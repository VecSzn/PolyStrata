# Groot Q4_K_M images on an RTX 5060 Laptop 8 GB

Image-path check for the Groot family's optional BF16 projector, 2026-11-03. One machine, one session; these are
measurements, not estimates.

- Model: `GrootTeam_Groot-19B-A3B-261023-Q4_K_M.gguf`, 12,484,339,584 bytes, SHA-256
  `c941e12a231b40b8fb9f6ad7f6601878d72db39c2c62aefb185fbe5c7d0c2e29`.
- Projector: `mmproj-GrootTeam_Groot-19B-A3B-261023-bf16.gguf`, 902,822,208 bytes, SHA-256
  `0140cef0473d63595a030c68b75b2c8d110bb623bb12ae5993c0cfdfb2d2b0a7` (the pinned revision; the file on disk
  matches). The projector declares `qwen3vl_merger`.
- Engine: the ready-made `strata-poly` 0.1.45 and `strata-vision` from the same bundle; branch `add-groot-family`.
- PC: RTX 5060 Laptop 8 GB, driver 617.42; Core Ultra 9 275HX, 15 threads; 32 GB of RAM; Windows 11.
- Settings: context 32,768, MTP on, `"vision": {"gpu": false, "max_tokens": 1024, "threads": 15}` — the CPU encoder
  that setup picks on an 8 GB card.
- Protocol: six fresh 768×512 PNGs, each with a three-letter-plus-two-digit code in black and a random number of red
  circles and blue squares; greedy, reasoning off, `max_tokens` 200; the answer counts as correct only when all three
  values match exactly: `CODE; circles; squares`.

| Run | Expected | Answer | Correct | Wall s | Prompt ms | Decode tok/s |
| ---: | --- | --- | :---: | ---: | ---: | ---: |
| 1 | CEF-30; 3; 2 | CEF-30; 3; 2 | yes | 5.49 | 592 | 105.3 |
| 2 | LLR-53; 5; 2 | LLR-53; 6; 2 | no | 4.20 | 497 | 84.7 |
| 3 | NMB-20; 5; 2 | NMB-20; 5; 2 | yes | 4.57 | 507 | 73.2 |
| 4 | SAX-85; 5; 2 | SAX-85; 5; 2 | yes | 4.71 | 518 | 86.1 |
| 5 | XEF-93; 4; 1 | XEF-93; 4; 1 | yes | 4.55 | 506 | 103.4 |
| 6 | VRZ-73; 4; 4 | VRZ-73; 4; 4 | yes | 4.58 | 496 | 95.4 |

5 of 6 answers were exactly right. The codes were read 6/6 and the square counts 6/6; the miss is one circle count
(run 2 answered 6 where the picture had 5). Counting small shapes is a known weak spot for vision models, so this is
a smoke test of the projector path, not a claim about accuracy on real images.

The server was ready 8.6 s after launch. A picture becomes 441 prompt tokens here (image embeddings plus the
question). The first request took 5.49 s of wall time because it starts the encoder; the next five took 4.2-4.7 s.
`prompt_ms` is the engine's read of the prompt with the image embeddings and does not include the CPU encode, so
client wall time minus `prompt_ms` is roughly the encode. Whole-machine VRAM was 6.9-7.1 GiB after each request — the
same range as the text-only run, because the encoder ran on the CPU. The six test images were generated for the
run and are not kept; each row above and its entry in `vision.json` describe one of them.

`vision.json` holds the same rows. Not checked: GPU image encoding, several pictures in one request, images with
conversation history, and other cards.
