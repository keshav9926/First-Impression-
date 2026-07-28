"""
ORIGIN: Manual creation to convert PRESENTATION.md into PowerPoint presentation.pptx
PURPOSE: Parse master slide deck markdown and generate a 16:9 dark-mode PowerPoint presentation (.pptx)
DESTINATION: Output saved as presentation.pptx in repository root
"""

import os
import re
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_presentation():
    md_path = "PRESENTATION.md"
    if not os.path.exists(md_path):
        print(f"Error: {md_path} not found.")
        return

    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by slide separators '---'
    raw_slides = [s.strip() for s in content.split("\n---\n") if s.strip()]

    prs = Presentation()
    # Set 16:9 Widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette (Dark Mode)
    BG_COLOR = RGBColor(15, 23, 42)        # Slate 900 #0F172A
    CARD_BG = RGBColor(30, 41, 59)         # Slate 800 #1E293B
    TITLE_COLOR = RGBColor(129, 140, 248)  # Indigo 400 #818CF8
    SUBTITLE_COLOR = RGBColor(148, 163, 184) # Slate 400 #94A3B8
    TEXT_COLOR = RGBColor(241, 245, 249)   # Slate 100 #F1F5F9
    ACCENT_COLOR = RGBColor(56, 189, 248)  # Sky 400 #38BDF8
    CODE_BG = RGBColor(15, 23, 42)         # Code dark bg

    for idx, slide_md in enumerate(raw_slides):
        slide = prs.slides.add_slide(blank_layout)

        # 1. Background Fill
        bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
        bg.fill.solid()
        bg.fill.fore_color.rgb = BG_COLOR
        bg.line.fill.background()

        # Check if title slide (first item with header block)
        lines = slide_md.split("\n")
        
        # Extract Slide Title & Subtitle/Body
        title_text = f"Slide {idx + 1}"
        body_lines = []

        for line in lines:
            if line.startswith("## Slide"):
                title_text = line.lstrip("#").strip()
            elif line.startswith("# ") and idx == 0:
                title_text = line.lstrip("#").strip()
            else:
                body_lines.append(line)

        # 2. Add Top Header Bar
        header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.733), Inches(0.9))
        tf_header = header_box.text_frame
        tf_header.word_wrap = True
        p_head = tf_header.paragraphs[0]
        p_head.text = title_text
        p_head.font.name = "Segoe UI"
        p_head.font.size = Pt(26)
        p_head.font.bold = True
        p_head.font.color.rgb = TITLE_COLOR

        # Add decorative accent line under header
        line_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.3), Inches(11.733), Inches(0.04))
        line_shape.fill.solid()
        line_shape.fill.fore_color.rgb = ACCENT_COLOR
        line_shape.line.fill.background()

        # 3. Body Content Container Box
        body_text = "\n".join(body_lines).strip()

        # If body contains a code block ```
        if "```" in body_text:
            # Separate text before code and code content
            parts = body_text.split("```")
            text_part = parts[0].strip()
            code_part = parts[1].strip() if len(parts) > 1 else ""

            # Left/Top text box
            if text_part:
                tb_text = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4))
                tf_t = tb_text.text_frame
                tf_t.word_wrap = True
                _add_formatted_text(tf_t, text_part, TEXT_COLOR, SUBTITLE_COLOR)

            # Code box on right/bottom
            code_x = Inches(6.6) if text_part else Inches(0.8)
            code_w = Inches(5.9) if text_part else Inches(11.733)
            
            code_bg_shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, code_x, Inches(1.5), code_w, Inches(5.4))
            code_bg_shape.fill.solid()
            code_bg_shape.fill.fore_color.rgb = CARD_BG
            code_bg_shape.line.color.rgb = ACCENT_COLOR

            tb_code = slide.shapes.add_textbox(code_x + Inches(0.2), Inches(1.6), code_w - Inches(0.4), Inches(5.2))
            tf_c = tb_code.text_frame
            tf_c.word_wrap = True
            
            # Clean first line of code block if language specified (e.g., ```python)
            code_lines = code_part.split("\n")
            if code_lines and code_lines[0].strip().isalpha():
                code_lines = code_lines[1:]
            
            p_code = tf_c.paragraphs[0]
            p_code.text = "\n".join(code_lines)
            p_code.font.name = "Consolas"
            p_code.font.size = Pt(11)
            p_code.font.color.rgb = RGBColor(226, 232, 240)

        else:
            # Full width text content
            content_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(11.733), Inches(5.4))
            tf_body = content_box.text_frame
            tf_body.word_wrap = True
            _add_formatted_text(tf_body, body_text, TEXT_COLOR, SUBTITLE_COLOR)

    output_pptx = "presentation.pptx"
    prs.save(output_pptx)
    print(f"Successfully generated {output_pptx} with {len(raw_slides)} slides.")

def _add_formatted_text(tf, raw_text, text_color, subtitle_color):
    lines = raw_text.split("\n")
    first = True
    for line in lines:
        sline = line.strip()
        if not sline:
            continue
        
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False

        if sline.startswith("### "):
            p.text = sline.lstrip("### ").strip()
            p.font.name = "Segoe UI"
            p.font.size = Pt(20)
            p.font.bold = True
            p.font.color.rgb = subtitle_color
            p.space_after = Pt(8)
        elif sline.startswith("#### "):
            p.text = sline.lstrip("#### ").strip()
            p.font.name = "Segoe UI"
            p.font.size = Pt(16)
            p.font.bold = True
            p.font.color.rgb = subtitle_color
            p.space_after = Pt(6)
        elif sline.startswith("- ") or sline.startswith("* "):
            p.text = "• " + sline[2:].replace("**", "")
            p.font.name = "Segoe UI"
            p.font.size = Pt(14)
            p.font.color.rgb = text_color
            p.space_after = Pt(4)
        elif sline.startswith("> "):
            p.text = sline.lstrip("> ").strip()
            p.font.name = "Segoe UI"
            p.font.size = Pt(14)
            p.font.italic = True
            p.font.color.rgb = subtitle_color
            p.space_after = Pt(6)
        else:
            p.text = sline.replace("**", "")
            p.font.name = "Segoe UI"
            p.font.size = Pt(14)
            p.font.color.rgb = text_color
            p.space_after = Pt(4)

if __name__ == "__main__":
    create_presentation()
