# Generate corrected shot 16 - fixes perspective/scale between Kess and Liliane
# Uses Gemini 2.5 Flash image generation with character reference images
#
# pip install google-genai

import base64
import mimetypes
import os
from google import genai
from google.genai import types


def load_image_part(file_path):
    """Load an image file as a Gemini Part."""
    with open(file_path, "rb") as f:
        data = f.read()
    mime_type = mimetypes.guess_type(file_path)[0] or "image/jpeg"
    return types.Part.from_bytes(data=data, mime_type=mime_type)


def save_binary_file(file_name, data):
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"File saved to: {file_name}")


def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    # Load character and location reference images
    kess_ref = load_image_part("Characters/Kess_BravoSierra/Kess_BravoSierra_reference_neutral_frontal_001.jpg")
    liliane_ref = load_image_part("Characters/Liliane_FoxKilo/Liliane_FoxKilo_reference_neutral_frontal_001.jpg")
    admin_hub_ref = load_image_part("Locations/Admin_Hub/Admin_Hub_cinematic_001.jpg")

    prompt = """Generate a sgraffito industrial expressionism illustration in 16:9 widescreen landscape format.

SCENE: Immediately after an assassination attempt on the governor during a public address. A Deployed Admin security officer kneels beside the wounded governor on the ground, urgently speaking into a handheld radio. Panicking crowd scatters in the background.

COMPOSITION - CRITICAL: Both characters must be on the SAME spatial plane at a SIMILAR distance from the camera. Use a MEDIUM SHOT framing that shows both characters from roughly the waist up. The man (kneeling) and the woman (on the ground) should appear at CORRECT PROPORTIONAL SCALE to each other. Do NOT use forced perspective or extreme foreground/background separation between them. They are right next to each other.

CHARACTER 1 - The man (reference image 1 - Kess):
- Athletic build, 42 years old, short dark hair graying at temples, warm green eyes, small scar on chin
- Dark blue Deployed Admin uniform with silver trim, duty belt
- KNEELING beside the fallen woman, one hand holding radio to mouth, other hand reaching toward her
- Expression: urgent, protective, alarmed
- He is NOT standing over her from a distance - he is RIGHT BESIDE her at ground level

CHARACTER 2 - The woman (reference image 2 - Liliane):
- Slender build, appears 37, dark brown hair in strict bun, hazel eyes, pale skin
- Gray bureaucratic jacket with rhodium insignia pin, FK-382 tattoo on wrist
- Lying on the ground, wounded (shoulder area), conscious but in pain
- She should appear the SAME SCALE as the man - they are next to each other

BACKGROUND:
- Brutalist government architecture with rhodium silver and cold blue accents (reference image 3)
- Black starry sky, no atmospheric haze
- Panicking crowd figures fleeing in the background at CORRECT smaller scale (they are farther away)
- Harsh artificial lighting, stark shadows with near-zero gradient

STYLE:
- Sgraffito technique: thick impasto paint layers scraped away revealing metallic textures
- Bold confident brushstrokes with aggressive energy
- High contrast, graphic quality
- Industrial grays with cold blue and rhodium silver accents
- Metallic undertones throughout
- Strong silhouettes
- Dark, gritty, oppressive atmosphere"""

    contents = [
        types.Content(
            role="user",
            parts=[
                kess_ref,
                types.Part.from_text(text="Reference image 1 - Kess (the Deployed Admin security officer): athletic man, short dark hair, green eyes, blue uniform"),
                liliane_ref,
                types.Part.from_text(text="Reference image 2 - Liliane (the wounded governor): slender woman, dark brown hair in bun, gray jacket with rhodium pin"),
                admin_hub_ref,
                types.Part.from_text(text="Reference image 3 - Admin Hub architecture (background reference)"),
                types.Part.from_text(text=prompt),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        response_modalities=[
            "IMAGE",
            "TEXT",
        ],
    )

    model = "gemini-2.5-flash-image-preview"
    output_path = "storyboards/shot_16_first.png"

    print(f"Generating corrected shot 16...")
    file_index = 0
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue
        if chunk.candidates[0].content.parts[0].inline_data and chunk.candidates[0].content.parts[0].inline_data.data:
            inline_data = chunk.candidates[0].content.parts[0].inline_data
            data_buffer = inline_data.data
            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            save_path = output_path if file_index == 0 else f"storyboards/shot_16_alt_{file_index}{file_extension}"
            save_binary_file(save_path, data_buffer)
            file_index += 1
        else:
            print(chunk.text)

    if file_index > 0:
        print(f"\nDone! Generated {file_index} image(s).")
        print(f"Primary output: {output_path}")
        print(f"\nNext step: regenerate the video from the new image using your video generation tool.")
    else:
        print("No image was generated. Check the prompt or API response.")


if __name__ == "__main__":
    generate()
