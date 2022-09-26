# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import json
import os
from glob import glob
from re import L

import openai
from cog import BasePredictor, Input, Path
from dotenv import load_dotenv
from googletrans import Translator

load_dotenv()
from time import sleep

from pypollsdk import Model

openai.api_key = os.getenv("OPENAI_API_KEY")


gpt_prompt = """Prompt Design

You can borrow some photographic prompt terminology (especially for framing) to apply to illustrations: e.g: 'close-up.' If you are generating mockups of 3D art, you can also define how that piece is photographed!

Adjectives can easily influence multiple factors, e.g: 'art deco' will influence the illustration style, but also the clothing and materials of the subject, unless otherwise defined. Years, decades and eras, like '1924' or 'late-90s', can also have this effect.

Even superficially specific prompts have more 'general' effects. For instance, defining a camera or lens ('Sigma 75mm') doesn't just 'create that specific look', it more broadly alludes to 'the kind of photo where the lens/camera appears in the description', which tend to be professional and hence higher-quality.

If a style is proving elusive, try 'doubling down' with related terms (artists, years, media, movement) years, e.g: rather than simply '…by Picasso' , try '…Cubist painting by Pablo Picasso, 1934, colourful, geometric work of Cubism

Detailed prompts are great if you know exactly what you're looking for and are trying to get a specific effect. …but DALL·E also has a creative eye, and has studied over 400 million images. So there is nothing wrong with being vague, and seeing what happens! You can also use variations to create further riffs of your favourite output. Sometimes you'll end up on quite a journey!

If the prompt is already very detailed, there is no need to add much more.

Examples of how to make prompts

prompt: A universe in a jar
pimped: Intricate illustration of a universe in a jar. intricately exquisitely detailed. holographic. beautiful. colourful. 3 d vray render, artstation, deviantart, pinterest, 5 0 0 px models

prompt: A fox wearing a cloak
pimped: A fox wearing a cloak. angled view, cinematic, mid-day, professional photography,8k, photo realistic, 50mm lens , Pixar, Dreamworks, Alex Ross, Tim Burton, Nickelodeon, Alex Ross, Character design, breath of wild, 3d render
 
prompt: Jellyfish phoenix goddess
pimped: Goddess portrait. jellyfish phoenix head. intricate artwork by tooth wu and wlop and beeple. octane render, trending on artstation, greg rutkowski very coherent symmetrical artwork. cinematic, hyper realism, high detail, octane render, 8k

prompt: Cute humanoid robot portrait
pimped: Cute humanoid robot, crystal material, portrait bust, symmetry, faded colors, aztec theme, cypherpunk background, tim hildebrandt, wayne barlowe, bruce pennington, donato giancola, larry elmore, masterpiece, trending on artstation, featured on pixiv, cinematic composition, beautiful lighting, hyper detailed, 8 k, unreal engine 5

prompt: Portrait of Harry Potter
pimped: A close up portrait of harry potter as a young man, art station, highly detailed, focused gaze, concept art, sharp focus, illustration in pen and ink, wide angle, by kentaro miura

prompt: Artichoke head monster
pimped: Humanoid figure with an artichoke head, highly detailed, digital art, sharp focus, trending on art station, monster, glowing eyes, anime art style

prompt: A boy and a girl with black short hair in a rowing boat
pimped: A boy and a girl with long flowing auburn hair sitting together on the rowboat. boy has black short hair, boy has black short hair. atmospheric lighting, long shot, romantic, boy and girl are the focus, trees, river. details, sharp focus, illustration, by jordan grimmer and greg rutkowski, trending artstation, pixiv, digital art

prompt: Diagram of a bird
pimped: A patent drawing of mechanical bird robots, birds of all kinds, infographic, intricate drawing, 1960s advertising, watercolour, ink drawing, patent drawing, wireframe, technical and mechanical details, descriptions, explosion drawing, cad

prompt: Coffee grinder
pimped: drawn coffee grinder in the style of thomas edison, patent filing, detailed, hd

prompt: {}
pimped:""".format


def report_status(**kwargs):
    status = json.dumps(kwargs)
    print(f"pollen_status: {status}")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # self.model = torch.load("./weights.pth")
        self.stable_diffusion = Model(
            "614871946825.dkr.ecr.us-east-1.amazonaws.com/pollinations/stable-diffusion-private"
        )
        self.translator= Translator()

    def predict(
        self,
        prompt: str,
    ) -> Path:
        """Run a single prediction on the model"""

        # JSON encode {title: "Pimping your prompt", payload: prompt }
        report_status(title="Translating to English", payload=prompt)
        prompt = self.translator.translate(prompt.strip()).text 
        report_status(title="Pimping prompt", payload=prompt)
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=gpt_prompt(prompt),
            max_tokens=200,
            temperature=0.82,
            n=3,
            stop=["prompt:", "\n"]
        ).choices
        prompts = [i.text.strip().replace("pimped: ", "") for i in response]
        report_status(title="Generating images", payload="\n".join(prompts))

        prompts = "\n".join(prompts)
        print("prompts:", prompts)

        self.stable_diffusion.predict({"prompts": prompts, "num_frames_per_prompt": 1, "diffusion_steps": -50, "prompt_scale": 15}, "/outputs/stable-diffusion")
        report_status(title="Display", payload=prompts)
        os.system("mv -v /outputs/stable-diffusion/*.png /outputs")
        sleep(5)
        return
