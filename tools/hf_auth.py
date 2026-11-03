"""Session-only Hugging Face authentication shared by setup and its download helpers."""
import argparse
import os
import urllib.error
import urllib.parse
import urllib.request


class HFAuthError(RuntimeError):
    """An access failure that retrying the same credentials cannot fix."""


def token_value(value):
    value = value.strip()
    if not value or any(ord(c) <= 32 or ord(c) >= 127 for c in value):
        raise argparse.ArgumentTypeError("use a non-empty Hugging Face token without whitespace")
    return value


def add_argument(parser):
    parser.add_argument("--hf-token", type=token_value, metavar="TOKEN",
                        help="Hugging Face download token for this run (overrides HF_TOKEN; not saved)")


def configure(parser, args):
    # Remove the credential from the settings namespace as soon as it is consumed.
    explicit = vars(args).pop("hf_token", None)
    value = explicit if explicit is not None else os.environ.get("HF_TOKEN")
    if value:
        try:
            os.environ["HF_TOKEN"] = token_value(value)
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))


def trusted(url):
    try:
        parts = urllib.parse.urlsplit(url)
        return (parts.scheme == "https" and parts.hostname == "huggingface.co"
                and parts.port in (None, 443) and parts.username is None and parts.password is None)
    except ValueError:
        return False


class _Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        redirected = super().redirect_request(req, fp, code, msg, headers, newurl)
        if redirected is not None:
            if req.get_method() == "HEAD":
                redirected.method = "HEAD"
            redirected.remove_header("Authorization")
            # Authorization is never copied to a CDN, mirror, or HTTP redirect.
            if trusted(req.full_url) and trusted(newurl):
                auth = req.get_header("Authorization")
                if auth:
                    redirected.add_unredirected_header("Authorization", auth)
        return redirected


def urlopen(req, timeout=60):
    token = os.environ.get("HF_TOKEN")
    try:
        if token and trusted(req.full_url):
            req.add_unredirected_header("Authorization", "Bearer " + token_value(token))
            return urllib.request.build_opener(_Redirect()).open(req, timeout=timeout)
        # Preserve the existing anonymous downloader, including test/local mirror support.
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403) and trusted(error.url):
            error.close()
            raise HFAuthError(
                "Hugging Face denied access (HTTP %s). Check --hf-token or HF_TOKEN, token read permissions, "
                "and access to the repository (including any gated-model agreement)." % error.code
            ) from None
        raise
