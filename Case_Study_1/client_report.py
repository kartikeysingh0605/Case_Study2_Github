"""Presentation-only client inspection report.

Consumes an already-written Stage 7E JSON report and normalized image. It DOES NOT
recompute detector features, modify component decisions, or claim that an
out-of-distribution assembly has been reliably identified.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Mapping

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1920, 2410
NAVY = '#14263A'
NAVY2 = '#22394F'
BG = '#F1F5F9'
WHITE = '#FFFFFF'
TEXT = '#172A3A'
MUTED = '#617388'
LINE = '#DAE3EC'
GREEN = '#15803D'
GREEN_BG = '#E9F8EE'
RED = '#B42330'
RED_BG = '#FDEDEF'
AMBER = '#9A6204'
AMBER_BG = '#FFF5DB'
BLUE = '#2563EB'


def font(size: int, bold: bool = False):
    font_paths = (
        [r'C:\Windows\Fonts\seguisb.ttf', r'C:\Windows\Fonts\arialbd.ttf',
         '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']
        if bold else
        [r'C:\Windows\Fonts\segoeui.ttf', r'C:\Windows\Fonts\arial.ttf',
         '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']
    )
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text(draw: ImageDraw.ImageDraw, xy, value, size=23, color=TEXT, bold=False):
    draw.text(xy, str(value), fill=color, font=font(size, bold))


def fits(draw: ImageDraw.ImageDraw, value: str, max_width: int, size: int, bold=False):
    f = font(size, bold)
    if draw.textbbox((0, 0), value, font=f)[2] <= max_width:
        return value
    while value and draw.textbbox((0, 0), value + '…', font=f)[2] > max_width:
        value = value[:-1]
    return value.rstrip() + '…'


def wrap(draw: ImageDraw.ImageDraw, value: str, max_width: int, size=20, bold=False):
    f = font(size, bold)
    lines = []
    current = ''
    for word in value.split():
        candidate = word if not current else current + ' ' + word
        if draw.textbbox((0, 0), candidate, font=f)[2] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or ['']


def round_box(draw: ImageDraw.ImageDraw, box, fill=WHITE, radius=18, outline=None, width=1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def result_class(status: str):
    if status == 'POSSIBLY PRESENT':
        return 'PASS', GREEN, GREEN_BG
    if status.startswith('SUSPECT'):
        return 'SUSPECT', RED, RED_BG
    return 'REVIEW', AMBER, AMBER_BG


def compatibility_screen(results: list[dict]):
    """A conservative presentation-only WARNING, not a validated OOD classifier.

    An unrelated gearbox often causes many simultaneous mismatches across
    mechanical supports and major gears. An extensively damaged compatible
    gearbox may also do so. Hence this is a reason to review, never proof.
    """
    suspects = [r for r in results if r.get('status', '').startswith('SUSPECT')]
    support_count = sum('bearing' in r.get('component', '').lower() or
                        'Motor mounting plate' in r.get('component', '')
                        for r in suspects)
    gear_count = sum(r.get('component') in
                     ('Brass output gear', 'Idler gear', 'Compound gear', 'Motor pinion gear')
                     for r in suspects)
    if len(suspects) >= 8 and support_count >= 3 and (gear_count >= 1 or len(suspects) >= 12):
        return {
            'status': 'POSSIBLE LAYOUT MISMATCH',
            'flagged': True,
            'reason': f'{len(suspects)} suspected parts, including {support_count} structural support checks; the image may differ from the reference layout or have extensive damage.',
            'limits': 'Heuristic warning only. Not a validated unsupported-image detector.'
        }
    return {
        'status': 'NOT INDEPENDENTLY VERIFIED',
        'flagged': False,
        'reason': 'Blue plate alignment succeeded. Matching the complete mechanical layout has not been independently certified.',
        'limits': 'A different assembly can still produce a seemingly valid component result.'
    }


def normalized_marked_image(normalized_path: Path, rows: list[dict], geometry: Mapping):
    source = Image.open(normalized_path).convert('RGB')
    base = source.copy()
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    index_font = font(15, True)
    for index, result in enumerate(rows, 1):
        loc = geometry.get(result['component'])
        if not loc:
            continue
        typ, cx, cy, radius = loc
        _, color, _ = result_class(result['status'])
        if typ == 'circle':
            region = [cx-radius, cy-radius, cx+radius, cy+radius]
            od.ellipse(region, outline=color, width=4)
            ix, iy = cx + radius - 8, cy - radius - 4
        else:
            region = [cx-radius, cy-radius, cx+radius, cy+radius]
            od.rectangle(region, outline=color, width=4)
            ix, iy = cx - radius - 8, cy - radius - 4
        ix = max(18, min(base.width-20, ix))
        iy = max(18, min(base.height-20, iy))
        od.ellipse((ix-16, iy-16, ix+16, iy+16), fill=color, outline='white', width=2)
        label = str(index)
        bb = od.textbbox((0,0),label,font=index_font)
        od.text((ix-(bb[2]-bb[0])/2,iy-(bb[3]-bb[1])/2-3), label, font=index_font, fill='white')
    return Image.alpha_composite(base.convert('RGBA'), overlay).convert('RGB')


def format_score(value):
    return '—' if value is None else f'{float(value):.3f}'


def small_legend(d, x, y, c, label):
    d.ellipse([x, y+3, x+19, y+22], fill=c)
    text(d, (x+29,y-1), label, 19, MUTED)


def create_report(json_path, normalized_path, geometry=None):
    json_path = Path(json_path)
    normalized_path = Path(normalized_path)
    data = json.loads(json_path.read_text(encoding='utf-8'))
    results = data['component_results']
    report_name = Path(data['image']).stem
    image_name = Path(data['image']).name
    compat = compatibility_screen(results)
    flagged = compat['flagged']
    raw_verdict = data['verdict']
    presentation_verdict = 'MANUAL REVIEW' if flagged else raw_verdict
    count_pass = sum(r['status'] == 'POSSIBLY PRESENT' for r in results)
    count_suspect = sum(r['status'].startswith('SUSPECT') for r in results)
    count_review = len(results)-count_pass-count_suspect

    image = Image.new('RGB', (WIDTH, HEIGHT), BG)
    d = ImageDraw.Draw(image)
    d.rectangle((0,0, WIDTH,173), fill=NAVY)
    text(d, (67,43), 'GEAR ASSEMBLY', 40, WHITE, True)
    text(d, (69,101), 'VISUAL INSPECTION  /  COMPONENT VERIFICATION', 22, '#B8C9D9')
    text(d, (1150,51), fits(d, image_name, 700, 27), 27, WHITE, True)
    text(d, (1152,105), 'Reference-based analysis  •  CPU', 20, '#B8C9D9')

    # Four clean summary tiles.
    tiles=[('INSPECTION RESULT', presentation_verdict),
           ('COMPONENT CHECKS', str(len(results))),
           ('SUSPECTED DEFECTS', str(count_suspect)),
           ('REVIEW CHECKS', str(count_review))]
    colors=[AMBER if flagged or raw_verdict == 'REVIEW REQUIRED' else RED if raw_verdict == 'NOT OK' else GREEN,
            NAVY, RED if count_suspect else GREEN, AMBER if count_review else GREEN]
    for i,(label,value) in enumerate(tiles):
        x=64 + i*465
        round_box(d, (x,209,x+442,331), WHITE, 16, LINE, 1)
        text(d, (x+26,230), label, 17, MUTED, True)
        text(d, (x+26,265), fits(d,value,390,29,True), 29, colors[i], True)

    # Image card
    round_box(d, (64,364,1276,1225), WHITE, 18, LINE)
    text(d,(87,388),'ALIGNED IMAGE  /  NUMBERED INSPECTION REGIONS',21,NAVY,True)
    marked=normalized_marked_image(normalized_path,results,geometry or {})
    marked.thumbnail((1130,710),Image.Resampling.LANCZOS)
    image.paste(marked, (90+(1156-marked.width)//2,438+(710-marked.height)//2))
    d=ImageDraw.Draw(image)
    small_legend(d, 105,1180,GREEN,'Possibly present')
    small_legend(d, 365,1180,RED,'Suspect')
    small_legend(d, 565,1180,AMBER,'Review / unassessable')

    # Sidebar
    sx=1300
    round_box(d,(sx,364,1856,1225),WHITE,18,LINE)
    text(d,(sx+28,395),'INSPECTION ASSESSMENT',22,NAVY,True)
    y=442
    status_col=AMBER if flagged else RED if raw_verdict=='NOT OK' else GREEN if raw_verdict=='OK' else AMBER
    status_bg=AMBER_BG if flagged else RED_BG if raw_verdict=='NOT OK' else GREEN_BG if raw_verdict=='OK' else AMBER_BG
    round_box(d,(sx+27,y,sx+528,y+99),status_bg,12)
    text(d,(sx+47,y+14),'Report disposition',16,MUTED)
    text(d,(sx+47,y+42),fits(d,presentation_verdict,460,27,True),27,status_col,True)
    y+=127
    text(d,(sx+29,y),'Automated detector verdict',17,MUTED)
    text(d,(sx+29,y+29),raw_verdict,24,TEXT,True)
    y+=87
    d.line([(sx+27,y),(sx+529,y)], fill=LINE, width=2)
    y+=27
    text(d,(sx+29,y),'REFERENCE-LAYOUT CHECK',18,NAVY,True)
    y+=34
    for line in wrap(d, compat['status'],485,20,True):
        text(d,(sx+29,y),line,20,AMBER if flagged else MUTED,True); y+=28
    y+=9
    for line in wrap(d,compat['reason'],488,17):
        text(d,(sx+29,y),line,17,MUTED); y+=27
    y+=18
    d.line([(sx+27,y),(sx+529,y)],fill=LINE,width=2)
    y+=25
    text(d,(sx+29,y),'SUSPECTED COMPONENTS',19,NAVY,True)
    y+=42
    suspects=data.get('suspected_parts',[])
    if not suspects:
        text(d,(sx+29,y),'None flagged by the current checks.',18,GREEN)
        y+=31
    else:
        for n,part in enumerate(suspects[:7],1):
            s=f'{n}. {part}'
            text(d,(sx+29,y), fits(d,s,490,18),18,RED)
            y+=33
        if len(suspects)>7:
            text(d,(sx+29,y),f'+ {len(suspects)-7} more in component table below',17,MUTED)
            y+=32
    y=min(y+16,1126)
    if data.get('conditional_ok'):
        for line in wrap(d,'CONDITIONAL OK — hidden motor body inferred from visible proxies.',488,17):
            text(d,(sx+29,y),line,17,AMBER); y+=25

    # Components table
    round_box(d,(64,1252,1856,2170),WHITE,18,LINE)
    text(d,(91,1286),'COMPONENT VALIDATION',29,NAVY,True)
    text(d,(1340,1292),'Scores are feature measurements, not probabilities',17,MUTED)
    entries_per_col=(len(results)+1)//2
    col_w=858
    row_h=52
    header_y=1365
    for col in range(2):
        cx=92 + col*886
        d.rectangle((cx,header_y,cx+col_w,header_y+48),fill='#E8EFF7')
        text(d,(cx+13,header_y+12),'#  COMPONENT',18,NAVY,True)
        text(d,(cx+515,header_y+12),'SCORE',17,NAVY,True)
        text(d,(cx+632,header_y+12),'LIMIT',17,NAVY,True)
        text(d,(cx+738,header_y+12),'STATUS',17,NAVY,True)
        for k in range(entries_per_col):
            i=col*entries_per_col+k
            if i>=len(results):
                break
            row=results[i]
            y=header_y+49+k*row_h
            if k%2:
                d.rectangle((cx,y,cx+col_w,y+row_h-2), fill='#F8FAFC')
            nm=f'{i+1:02d}  {row["component"]}'
            text(d,(cx+12,y+14),fits(d,nm,486,19),19,TEXT)
            text(d,(cx+524,y+14),format_score(row.get('test_score')),19,TEXT)
            text(d,(cx+637,y+14),format_score(row.get('threshold')),19,TEXT)
            label, color, bcolor=result_class(row.get('status',''))
            round_box(d,(cx+741,y+10,cx+846,y+42),bcolor,8)
            text(d,(cx+747,y+15),label,16,color,True)

    # Footer / limitations
    text(d,(80,2202),'INTERPRETATION & LIMITATIONS',19,NAVY,True)
    lines=[
        'Green = possibly present; red = suspected missing/misplaced; amber = requires review. Numbers match image regions.',
        'A threshold is a detector feature limit, not a confidence percentage. No independent real-world accuracy is claimed.',
        'This report is an experimental visual aid. Confirm critical defects manually; keep the raw JSON for audit.',
    ]
    if flagged:
        lines.insert(0,'WARNING: widespread differences suggest a possible reference-layout mismatch or major damage; the detector verdict is not independently reliable.')
    if data.get('conditional_ok'):
        lines.insert(0,'CONDITIONAL OK: the hidden motor body is NOT seen by this camera and is inferred from visible parts.')
    y=2235
    for item in lines:
        for line in wrap(d,item,1750,18):
            text(d,(81,y),line,18,MUTED)
            y+=27
    text(d,(80,2363),f'Input: {image_name}     |     Policy: {data.get("decision_policy", "unknown")}     |     Processing: {data.get("seconds", "?")} s',18,NAVY)
    text(d,(1540,2363),'VISION INSPECTION  /  REVIEW COPY',16,MUTED,True)

    target=json_path.parent / f'{report_name}_client_report.png'
    image.save(target,optimize=True)
    raw_summary = {
        'image':image_name,
        'automated_detector_verdict':raw_verdict,
        'client_report_disposition':presentation_verdict,
        'layout_screening':compat,
        'component_checks':len(results),
        'pass_count':count_pass,
        'suspected_count':count_suspect,
        'review_count':count_review,
        'unchanged_json_report':str(json_path),
        'client_report_png':str(target),
        'note':'Client disposition is a presentation-level warning only; underlying inspect.py decisions are unchanged.'
    }
    summary_target=json_path.parent / f'{report_name}_client_summary.json'
    summary_target.write_text(json.dumps(raw_summary,indent=2),encoding='utf-8')
    return target, raw_summary
