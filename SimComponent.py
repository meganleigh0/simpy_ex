# ---- Slide 1: Title slide (robust - works with any GDLS theme) ----
title_layout = prs.slide_layouts[0]  # usually the title slide layout
title_slide = prs.slides.add_slide(title_layout)

# Set slide TITLE
if title_slide.shapes.title:
    title_slide.shapes.title.text = PROGRAM
else:
    # fallback - create a title textbox
    tx = title_slide.shapes.add_textbox(Inches(1), Inches(0.7), Inches(8), Inches(1))
    p = tx.text_frame.paragraphs[0]
    p.text = PROGRAM
    p.font.bold = True
    p.font.size = Pt(40)

# Try to find subtitle placeholder (index varies by theme)
subtitle_ph = None
for shp in title_slide.placeholders:
    if shp.placeholder_format.type == 1:  # SUBTITLE type
        subtitle_ph = shp
        break

if subtitle_ph:
    subtitle_ph.text = SNAPSHOT_DATE.strftime("%Y-%m-%d")
else:
    # No subtitle placeholder → add a textbox manually
    tx = title_slide.shapes.add_textbox(Inches(1), Inches(1.8), Inches(8), Inches(0.8))
    p = tx.text_frame.paragraphs[0]
    p.text = SNAPSHOT_DATE.strftime("%Y-%m-%d")
    p.font.size = Pt(24)