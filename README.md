# ComicCraft - AI Comic Generator

Home form -> Gemini writes story -> images per panel -> preview page -> PDF download -> success page.

## Run
1. `python -m venv venv` then activate (Windows `venv\Scripts\activate`, Mac/Linux `source venv/bin/activate`)
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`, paste key (`AIza...` from https://aistudio.google.com/apikey)
4. `python app.py` and open http://127.0.0.1:8000

## Check setup
- http://127.0.0.1:8000/api/health  (has_key must be true)
- http://127.0.0.1:8000/test-image  (shows which image backend works)
- http://127.0.0.1:8000/docs        (endpoint list)

## Notes
- Keep `gemini-flash-latest` (free tier). Pro models have no free quota.
- IMAGE_BACKEND: `pollinations` (free real images, default), `hf_api` (needs HF_TOKEN), `placeholder` (offline).
  If a real backend fails, drawn placeholder images are used so the app never crashes.
- No API key = demo story. Never upload `.env` to GitHub.
