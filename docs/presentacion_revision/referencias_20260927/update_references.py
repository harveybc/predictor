"""Citation-only OOXML edit; preserve media and every unrelated slide/shape."""
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib
import json
import re

from lxml import etree as E
from PIL import ImageFont

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "antes_referencias.pptx"
OUTPUT = HERE / "presentacion_entrevista_postulacion_doctoral_revisada.pptx"
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
EMU = 914400
with ZipFile(SOURCE) as z:
    infos = z.infolist()
    before = {i.filename: z.read(i.filename) for i in infos}
after = dict(before)
changes = []
references = json.loads((HERE / "references.json").read_text())


def xml_bytes(root):
    return E.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True)


def shape(root, identifier):
    found = root.xpath(f'.//p:sp[p:nvSpPr/p:cNvPr[@id="{identifier}"]]', namespaces=NS)
    assert len(found) == 1, identifier
    return found[0]


def replace(sp, old, new):
    nodes = sp.findall('.//a:t', NS)
    text = ''.join(n.text or '' for n in nodes)
    assert text.count(old) == 1, (old, text)
    start, end = text.index(old), text.index(old) + len(old)
    offset = 0
    inserted = False
    for node in nodes:
        value = node.text or ''
        stop = offset + len(value)
        if offset < end and stop > start:
            prefix = value[:max(0, start - offset)]
            suffix = value[max(0, end - offset):]
            node.text = prefix + (new if not inserted else '') + suffix
            inserted = True
        offset = stop
    assert ''.join(n.text or '' for n in nodes) == text[:start] + new + text[end:]


def replace_all(sp, text):
    old = ''.join(sp.xpath('.//a:t/text()', namespaces=NS))
    replace(sp, old, text)


def textbox(root, identifier, x, y, w, h, text, size=11, color="526372", bold=False, link=None):
    tree = root.find('p:cSld/p:spTree', NS)
    sp = E.SubElement(tree, f'{{{NS["p"]}}}sp')
    nv = E.SubElement(sp, f'{{{NS["p"]}}}nvSpPr')
    E.SubElement(nv, f'{{{NS["p"]}}}cNvPr', id=str(identifier), name=f'Reference {identifier}')
    E.SubElement(nv, f'{{{NS["p"]}}}cNvSpPr', txBox="1")
    E.SubElement(nv, f'{{{NS["p"]}}}nvPr')
    pr = E.SubElement(sp, f'{{{NS["p"]}}}spPr')
    xf = E.SubElement(pr, f'{{{NS["a"]}}}xfrm')
    E.SubElement(xf, f'{{{NS["a"]}}}off', x=str(round(x*EMU)), y=str(round(y*EMU)))
    E.SubElement(xf, f'{{{NS["a"]}}}ext', cx=str(round(w*EMU)), cy=str(round(h*EMU)))
    E.SubElement(pr, f'{{{NS["a"]}}}noFill')
    body = E.SubElement(sp, f'{{{NS["p"]}}}txBody')
    E.SubElement(body, f'{{{NS["a"]}}}bodyPr', wrap="square", lIns="0", rIns="0", tIns="0", bIns="0")
    E.SubElement(body, f'{{{NS["a"]}}}lstStyle')
    for line in text.split('\n'):
        p = E.SubElement(body, f'{{{NS["a"]}}}p')
        pp = E.SubElement(p, f'{{{NS["a"]}}}pPr')
        E.SubElement(E.SubElement(pp, f'{{{NS["a"]}}}lnSpc'), f'{{{NS["a"]}}}spcPts', val=str(round(size*120)))
        r = E.SubElement(p, f'{{{NS["a"]}}}r')
        rp = E.SubElement(r, f'{{{NS["a"]}}}rPr', lang="es-CO", sz=str(round(size*100)), b=str(int(bold)))
        E.SubElement(E.SubElement(rp, f'{{{NS["a"]}}}solidFill'), f'{{{NS["a"]}}}srgbClr', val=color)
        E.SubElement(rp, f'{{{NS["a"]}}}latin', typeface="Arial")
        if link:
            E.SubElement(rp, f'{{{NS["a"]}}}hlinkClick', {f'{{{NS["r"]}}}id': link})
        E.SubElement(r, f'{{{NS["a"]}}}t').text = line
    return sp


def edit(slide, edits, additions=()):
    name = f'ppt/slides/slide{slide}.xml'
    root = E.fromstring(before[name])
    for identifier, old, new in edits:
        replace(shape(root, identifier), old, new)
        changes.append({"slide": slide, "shape": identifier, "old": old, "new": new})
    for args in additions:
        textbox(root, *args)
    # Existing shapes outside the citation whitelist must remain XML-identical.
    original = E.fromstring(before[name])
    allowed = {str(i) for i, _, _ in edits}
    for sp in original.findall('p:cSld/p:spTree/p:sp', NS):
        identifier = sp.find('p:nvSpPr/p:cNvPr', NS).get('id')
        if identifier not in allowed:
            assert E.tostring(sp) == E.tostring(shape(root, identifier)), (slide, identifier)
    assert [E.tostring(p) for p in original.findall('.//p:pic', NS)] == [
        E.tostring(p) for p in root.findall('.//p:pic', NS)]
    after[name] = xml_bytes(root)


edit(5, [(74, 'M.DS[1]', 'M.DS [1, fig. 43]'),
         (74, 'plazo[2]', 'plazo [2]'),
         (75, 'Figura 2. - Barrido de ruido a pronósticos ideales usados en estrategia heurística de trading basada en predicciones de corto y largo plazo[2]',
          'Figura 2. Barrido de ruido gaussiano sobre un oráculo direccional: resultados y datos históricos de heuristic-strategy [2].')])
edit(8, [(126, 'Fuentes: Hyndman y Athanasopoulos, FPP3, 3.ª ed., 2021, caps. 4 y 9; SciPy, scipy.signal.hilbert.',
          'Fuentes: perfil y estacionalidad [3]; ADF/KPSS [4]; Hilbert, correlación y coherencia [5].')])
edit(9, [], [(400, .8, 6.08, 11.7, .25, 'Antecedentes: perfiles estadísticos [6] y combinación de pronósticos basada en características [7].')])
edit(10, [], [(400, .8, 6.62, 11.7, .25, 'Fuentes: redes convolucionales y recurrentes [8]; atención multi-cabezal [9].')])
edit(11, [(195, '[1], [2]', '[6], [7]')])
edit(12, [(204, '[3]', '[10]')])
edit(13, [(232, '[4], [5]', '[11], [12]')])
edit(14, [(265, '[4], [5]', '[11], [12]'), (269, '[6]', '[13]')])


def wrap(text, size, inches):
    font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', round(size*10))
    lines, current = [], ''
    for word in text.split():
        candidate = current + (' ' if current else '') + word
        if font.getlength(candidate) > inches*72*10 and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


layout = []
for page, entries in [(16, references[:7]), (17, references[7:])]:
    root = E.fromstring(before['ppt/slides/slide16.xml'])
    tree = root.find('p:cSld/p:spTree', NS)
    for sp in list(tree.findall('p:sp', NS)):
        if int(sp.find('p:nvSpPr/p:cNvPr', NS).get('id')) >= 295:
            tree.remove(sp)
    replace_all(shape(root, 289), f'Referencias ({page-15}/2)')
    replace_all(shape(root, 294), str(page))
    rels = E.fromstring(before['ppt/slides/_rels/slide16.xml.rels'])
    y = 1.65
    for ref in entries:
        lines = wrap(ref['text'], 13.5, 11.0)
        height = len(lines) * 13.5 * 1.2 / 72 + .025
        url_lines = wrap(ref['link_text'], 9.5, 11.0)
        url_height = len(url_lines)*9.5*1.2/72 + .03
        identifier = 400 + ref['n']*3
        textbox(root, identifier, .78, y, .48, .3, f'[{ref["n"]}]', 14, '147D78', True)
        textbox(root, identifier+1, 1.34, y, 11.0, height, '\n'.join(lines), 13.5, '203344')
        rid = f'rIdRef{ref["n"]}'
        E.SubElement(rels, f'{{{REL}}}Relationship', Id=rid, Type=NS['r']+'/hyperlink', Target=ref['url'], TargetMode='External')
        textbox(root, identifier+2, 1.34, y+height, 11.0, url_height, '\n'.join(url_lines), 9.5, '147D78', link=rid)
        layout.append({"slide": page, "reference": ref['n'], "y": y, "height": height+url_height})
        y += height + url_height + .05
    assert y < 6.95, (page, y)
    after[f'ppt/slides/slide{page}.xml'] = xml_bytes(root)
    after[f'ppt/slides/_rels/slide{page}.xml.rels'] = xml_bytes(rels)

# Append only the second bibliography page to the package.
presentation = E.fromstring(before['ppt/presentation.xml'])
ids = presentation.find('p:sldIdLst', NS)
E.SubElement(ids, f'{{{NS["p"]}}}sldId', {'id': str(max(int(x.get('id')) for x in ids)+1), f'{{{NS["r"]}}}id': 'rIdReferences17'})
after['ppt/presentation.xml'] = xml_bytes(presentation)
rels = E.fromstring(before['ppt/_rels/presentation.xml.rels'])
E.SubElement(rels, f'{{{REL}}}Relationship', Id='rIdReferences17', Type=NS['r']+'/slide', Target='slides/slide17.xml')
after['ppt/_rels/presentation.xml.rels'] = xml_bytes(rels)
ct = E.fromstring(before['[Content_Types].xml'])
E.SubElement(ct, '{'+E.QName(ct).namespace+'}Override', PartName='/ppt/slides/slide17.xml',
             ContentType='application/vnd.openxmlformats-officedocument.presentationml.slide+xml')
after['[Content_Types].xml'] = xml_bytes(ct)

with ZipFile(OUTPUT, 'w', compression=ZIP_DEFLATED) as z:
    for info in infos:
        z.writestr(info, after[info.filename])
    for name in after.keys() - before.keys():
        z.writestr(name, after[name])
with ZipFile(OUTPUT) as z:
    assert z.testzip() is None
    assert all(z.read(n) == b for n, b in before.items() if n.startswith('ppt/media/'))
    for i in [1, 2, 3, 4, 6, 7, 15]:
        name = f'ppt/slides/slide{i}.xml'
        assert z.read(name) == before[name], name

citation_order = []
for i in range(1, 16):
    r = E.fromstring(after[f'ppt/slides/slide{i}.xml'])
    text = ' '.join(''.join(sp.xpath('.//a:t/text()', namespaces=NS)) for sp in r.findall('.//p:sp', NS))
    for match in re.finditer(r'\[(\d+)(?:, fig\. \d+)?\]', text):
        number = int(match.group(1))
        if number not in citation_order:
            citation_order.append(number)
assert citation_order == list(range(1, 14)), citation_order
report = {
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "output_sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
    "original_slides": 16, "output_slides": 17,
    "all_media_byte_identical": True,
    "unrelated_existing_shapes_unchanged": True,
    "untouched_slides_byte_identical": [1, 2, 3, 4, 6, 7, 15],
    "citation_first_appearance": citation_order,
    "changes": changes, "bibliography_layout": layout,
    "changed_parts": [n for n in before if after[n] != before[n]],
}
(HERE / 'VERIFICATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('changes','bibliography_layout')}, indent=2))
