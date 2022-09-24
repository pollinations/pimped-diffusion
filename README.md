# Pimped-diffusion
GPT-3 pimps your prompt and stable-diffusion makes an image.

# Dev
1. Add your environment variables to `.env`
2. `cog build -t pimped-diffusion`
3. `mkdir test-outputs`
4. `docker run -p 5000:5000 --mount type=bind,source=$(pwd)/test-outputs,target=/outputs pimped-diffusion`
5. `python client.py`