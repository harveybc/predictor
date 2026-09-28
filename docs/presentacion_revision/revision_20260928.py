from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy
from lxml import etree as E
import hashlib
import json

DOCS = Path(__file__).resolve().parents[1]
HERE = DOCS / 'presentacion_revision' / 'revision_20260928'
HERE.mkdir(exist_ok=True)
PPT = DOCS / 'presentacion_entrevista_postulacion_doctoral_revisada.pptx'
backup = HERE / 'original_usuario.pptx'
raw = PPT.read_bytes()
assert not backup.exists() or backup.read_bytes() == raw
backup.write_bytes(raw)
ns = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}
edits = {
275: 'Qué aportará esta investigación',
281: 'Un método para elegir cómo procesar cada grupo de series\ny comprobar cuándo esa elección mejora los resultados.',
282: 'Cómo elegir',
283: 'Relacionar las propiedades\nde las series con las\ntransformaciones y modelos\nque funcionan mejor.',
284: 'Cuánto mejora',
285: 'Medir la mejora frente\na los modelos de referencia,\nsu coste de entrenamiento\ny cuándo deja de ayudar.',
286: 'Cómo reutilizarlo',
287: 'Publicar código,\nconfiguraciones y resultados\npara repetir las pruebas\nen pronóstico y trading.',
289: 'Comprobar la mejora en datos no usados para elegir la configuración [13].\nDespués: evaluar si también ayuda preentrenar el núcleo temporal.'
}
with ZipFile(PPT) as z:
    infos = z.infolist()
    before = {i.filename: z.read(i.filename) for i in infos}
after = dict(before)
key = 'ppt/slides/slide15.xml'
root = E.fromstring(before[key])
for s in root.findall('.//p:sp', ns):
    ident = int(s.find('.//p:cNvPr', ns).get('id'))
    if ident not in edits:
        continue
    body = s.find('p:txBody', ns)
    paragraphs = body.findall('a:p', ns)
    template = paragraphs[0]
    props = template.find('.//a:rPr', ns)
    for p in paragraphs:
        body.remove(p)
    for line in edits[ident].split('\n'):
        p = E.SubElement(body, '{'+ns['a']+'}p')
        pp = template.find('a:pPr', ns)
        if pp is not None:
            p.append(deepcopy(pp))
        r = E.SubElement(p, '{'+ns['a']+'}r')
        if props is not None:
            rp = deepcopy(props)
            if ident in [283,285,287]:
                rp.set('sz', '1850')
            if ident == 289:
                rp.set('sz', '1700')
            if ident == 281:
                rp.set('sz', '2200')
            r.append(rp)
        E.SubElement(r, '{'+ns['a']+'}t').text = line
after[key] = E.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
assert [k for k in before if before[k] != after[k]] == [key]
out = HERE / PPT.name
with ZipFile(out, 'w') as z:
    for info in infos:
        z.writestr(info, after[info.filename])
citations = []
import re
for i in range(1,16):
    doc = E.fromstring(after[f'ppt/slides/slide{i}.xml'])
    text = ' '.join(''.join(s.xpath('.//a:t/text()',namespaces=ns)) for s in doc.findall('.//p:sp',ns))
    for n in re.findall(r'\[(\d+)(?:, fig\. \d+)?\]',text):
        if int(n) not in citations:
            citations.append(int(n))
assert citations == list(range(1,14)), citations
(HERE/'VERIFICATION.json').write_text(json.dumps({
    'source_sha256': hashlib.sha256(raw).hexdigest(),
    'output_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
    'changed_parts': [key], 'all_other_parts_identical': True,
    'slides_1_to_14_byte_identical': True, 'all_media_identical': True,
    'first_citation_order': citations, 'text_changes': edits
},ensure_ascii=False,indent=2))
print(out)
