"""Authentication tests with fake tokens and HTTP responses; no downloads or GPU required."""
import argparse
import contextlib
from email.message import Message
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import urllib.error
import urllib.request
import urllib.response

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup as S
from tools import hf_auth as H
import mtp_fetch as M
import glm_draft_fetch as G
from metal import setup_mac as Mac

URL = "https://huggingface.co/owner/model/resolve/main/weights.gguf"
TOKEN = "hf_test_only"


class Authentication(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_cli_forms_and_environment_precedence(self):
        for argv, expected in (([], "hf_environment"), (["--hf-token", TOKEN], TOKEN),
                               (["--hf-token=" + TOKEN], TOKEN)):
            os.environ["HF_TOKEN"] = "hf_environment"
            parser = argparse.ArgumentParser()
            H.add_argument(parser)
            args = parser.parse_args(argv)
            H.configure(parser, args)
            self.assertEqual(os.environ["HF_TOKEN"], expected)
            self.assertNotIn("hf_token", vars(args))

    def test_invalid_values_do_not_echo_credentials(self):
        for argv in (["--hf-token"], ["--hf-token="], ["--hf-token", TOKEN + "\ninvalid"]):
            parser = argparse.ArgumentParser()
            H.add_argument(parser)
            output = io.StringIO()
            with contextlib.redirect_stderr(output), self.assertRaises(SystemExit):
                parser.parse_args(argv)
            self.assertNotIn(TOKEN, output.getvalue())

    def test_authentication_is_only_for_official_https_origin(self):
        os.environ["HF_TOKEN"] = TOKEN
        for url, authenticated in ((URL, True), (URL.replace(".co/", ".co:443/"), True),
                (URL.replace("https:", "http:"), False),
                (URL.replace("huggingface.co", "huggingface.co.evil.test"), False),
                (URL.replace("huggingface.co", "hf-mirror.com"), False),
                (URL.replace("huggingface.co", "www.modelscope.cn"), False),
                (URL.replace("huggingface.co", "github.com"), False),
                (URL.replace("huggingface.co", "huggingface.co:8443"), False)):
            req = urllib.request.Request(url)
            with mock.patch.object(H.urllib.request, "build_opener") as build, \
                    mock.patch.object(H.urllib.request, "urlopen") as anonymous:
                H.urlopen(req)
                self.assertEqual(build.called, authenticated)
                self.assertEqual(anonymous.called, not authenticated)
                self.assertEqual(req.get_header("Authorization"), "Bearer " + TOKEN if authenticated else None)

    def test_redirects_keep_same_origin_auth_and_strip_cross_origin_auth(self):
        req = urllib.request.Request(URL, headers={"Range": "bytes=4-7"})
        req.add_unredirected_header("Authorization", "Bearer " + TOKEN)
        handler = H._Redirect()
        for destination, keep in ((URL + "?download=true", True),
                                   ("https://cdn.example/weights", False),
                                   (URL.replace("https:", "http:"), False)):
            redirected = handler.redirect_request(req, None, 302, "Found", {}, destination)
            self.assertEqual(redirected.get_header("Authorization"), "Bearer " + TOKEN if keep else None)
            self.assertEqual(redirected.get_header("Range"), "bytes=4-7")

    def test_access_failures_are_actionable_with_and_without_token(self):
        for token in ("", TOKEN):
            os.environ["HF_TOKEN"] = token
            for code in (401, 403):
                error = urllib.error.HTTPError(URL, code, "denied", {}, None)
                with mock.patch.object(H.urllib.request, "urlopen", side_effect=error), \
                        mock.patch.object(H.urllib.request, "build_opener") as build:
                    build.return_value.open.side_effect = error
                    with self.assertRaises(H.HFAuthError) as raised:
                        H.urlopen(urllib.request.Request(URL))
                self.assertIn("--hf-token or HF_TOKEN", str(raised.exception))
                self.assertNotIn(TOKEN, str(raised.exception))

    def test_real_opener_follows_redirects_without_leaking_auth_or_changing_head(self):
        os.environ["HF_TOKEN"] = TOKEN
        calls = []
        destinations = [URL + "?download=1", "https://cdn.example/weights", URL]

        class Transport(urllib.request.HTTPSHandler):
            def https_open(self, req):
                calls.append(req)
                headers = Message()
                if len(calls) <= 3:
                    headers["Location"] = destinations[len(calls) - 1]
                response = urllib.response.addinfourl(io.BytesIO(), headers, req.full_url,
                                                     302 if headers else 200)
                response.msg = "Found" if headers else "OK"
                return response

        opener = urllib.request.build_opener(H._Redirect(), Transport())
        with mock.patch.object(H.urllib.request, "build_opener", return_value=opener):
            H.urlopen(urllib.request.Request(URL, method="HEAD")).close()
        self.assertEqual([req.get_header("Authorization") for req in calls],
                         ["Bearer " + TOKEN, "Bearer " + TOKEN, None, None])
        self.assertEqual([req.get_method() for req in calls], ["HEAD"] * 4)

    def test_installer_access_failures_stop_before_retry_sleep(self):
        for during_get in (False, True):
            head = io.BytesIO()
            head.headers = {"Content-Length": "4"}
            error = H.HFAuthError("Hugging Face denied access; check HF_TOKEN")
            with tempfile.TemporaryDirectory() as folder, \
                    mock.patch.object(S.hf_auth, "urlopen", side_effect=[head, error] if during_get else error), \
                    mock.patch.object(S.time, "sleep") as sleep, \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit):
                S.download(URL, Path(folder) / "weights.gguf")
            sleep.assert_not_called()

    def test_resumed_download_authenticates_head_and_get_without_persisting_token(self):
        os.environ["HF_TOKEN"] = TOKEN
        calls = []

        def response(req, timeout):
            calls.append(req)
            result = io.BytesIO(b"data")
            result.headers = {"Content-Length": "8"}
            result.status = 206
            return result

        with tempfile.TemporaryDirectory() as folder, \
                mock.patch.object(H.urllib.request, "build_opener") as build, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            build.return_value.open.side_effect = response
            dst = Path(folder) / "weights.gguf"
            dst.with_suffix(".gguf.part").write_bytes(b"old-")
            S.download(URL, dst)
            self.assertEqual(dst.read_bytes(), b"old-data")
            self.assertEqual([r.get_method() for r in calls], ["HEAD", "GET"])
            self.assertEqual(calls[1].get_header("Range"), "bytes=4-")
            for req in calls:
                self.assertEqual(req.get_header("Authorization"), "Bearer " + TOKEN)
            for path in Path(folder).iterdir():
                self.assertNotIn(TOKEN.encode(), path.read_bytes())
            self.assertNotIn(TOKEN, output.getvalue())

    def test_draft_range_and_head_requests_authenticate(self):
        os.environ["HF_TOKEN"] = TOKEN
        calls = []

        def response(req, timeout):
            calls.append(req)
            result = io.BytesIO(b"data")
            result.headers = {"Content-Length": "4", "Content-Range": "bytes 0-3/4"}
            result.status = 206
            return result

        with mock.patch.object(H.urllib.request, "build_opener") as build:
            build.return_value.open.side_effect = response
            self.assertEqual(M.get(URL, 0, 3), b"data")
            self.assertEqual(G.size_of(URL), 4)
            M.resolve_repo()
        self.assertEqual([req.get_method() for req in calls], ["GET", "HEAD", "HEAD"])
        self.assertTrue(all(req.get_header("Authorization") == "Bearer " + TOKEN for req in calls))

    def test_draft_access_failure_is_not_retried(self):
        with mock.patch.object(M.hf_auth, "urlopen", side_effect=M.hf_auth.HFAuthError("denied")) as open_url, \
                mock.patch.object(M.time, "sleep") as sleep:
            with self.assertRaises(M.hf_auth.HFAuthError):
                M.get(URL)
            open_url.assert_called_once()
            sleep.assert_not_called()

    def test_mac_parser_consumes_token_without_warning_or_settings(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(Mac.main(["--hf-token", TOKEN, "--list-models"]), 0)
        self.assertEqual(os.environ["HF_TOKEN"], TOKEN)
        self.assertNotIn(TOKEN, output.getvalue())

    def test_linux_parser_sets_environment_before_backend_handoff(self):
        with mock.patch.object(sys, "platform", "linux"), \
                mock.patch.object(sys, "argv", ["setup.py", "--hf-token=" + TOKEN, "--backend", "sycl"]), \
                mock.patch.object(S, "sycl_setup", return_value=0) as handoff, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(S.main(), 0)
        handoff.assert_called_once()
        self.assertEqual(os.environ["HF_TOKEN"], TOKEN)
        self.assertNotIn(TOKEN, output.getvalue())

    def test_windows_wrappers_forward_arguments_without_echo(self):
        wrapper = (ROOT / "SETUP.bat").read_text()
        launcher = (ROOT / "START-HERE.bat").read_text()
        self.assertIn('--setup %*', wrapper)
        self.assertIn('setup.py %*', launcher)
        self.assertTrue(wrapper.startswith('@echo off'))
        self.assertTrue(launcher.startswith('@echo off'))


if __name__ == "__main__":
    unittest.main()
