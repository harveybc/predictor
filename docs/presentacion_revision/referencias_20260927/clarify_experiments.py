from pathlib import Path
from zipfile import ZipFile
from copy import deepcopy
from lxml import etree as E
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
PPT = ROOT / 'presentacion_entrevista_postulacion_doctoral_revisada.pptx'
HERE = Path(__file__).resolve().parent
BACKUP = HERE / 'antes_claridad_experimentos.pptx'
assert not BACKUP.exists() or BACKUP.read_bytes() == PPT.read_bytes(), 'Source changed.'
BACKUP.write_bytes(PPT.read_bytes())
NS = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'p': 'http://schemas.openxmlformats.org/presentationml/2006/main'}
changes = {
13: {
224: 'EXPERIMENTOS', 225: 'Qué vamos a probar y en qué datos',
231: 'Probar cada componente, después el sistema completo.',
232: 'Datos de electricidad, clima y tráfico de la literatura [11], [12], además de datos financieros.',
233: '1', 234: 'Señales controladas',
235: 'Crear series con ciclos, ruido y retardos.\nComparar cómo agruparlas y fusionarlas.',
236: '2', 237: 'Comparar alternativas',
238: 'Comparar modelos y transformaciones,\ncon y sin extractores preentrenados.',
239: '3', 240: 'Probar en otros datos',
241: 'Evaluar la solución elegida sin reajustarla\nen datos no usados para elegirla.',
242: '4', 243: 'Evaluar en trading',
244: 'Predecir precios y aprender decisiones.\nSimular costes y actualización semanal.',
245: 'Dos preguntas distintas: ¿mejora la predicción? ¿mejora también el resultado del trading?'
},
14: {
247: 'EVALUACIÓN', 248: 'Cómo sabremos si nuestra propuesta mejora',
254: 'Reproducir un modelo publicado y compararlo con nuestra propuesta.',
256: 'LOS MISMOS DATOS',
257: 'Mismas variables y fechas\npara entrenar y evaluar.',
259: 'LA MISMA PREDICCIÓN',
260: 'Mismo valor a predecir,\nhorizonte e información disponible.',
262: 'EL MISMO ERROR',
263: 'Misma fórmula y normalización\nque el estudio de referencia.',
265: 'Predicción [11], [12]',
266: 'Comparar MAE y MSE con el modelo publicado\ny con la predicción de referencia:\nque el valor actual no cambia.',
267: 'Trading y aprendizaje por refuerzo',
268: 'Comparar beneficio neto, Sharpe y caída máxima.\nUsar las mismas fechas, costes\ny reglas de ejecución.',
269: 'Elegir la configuración sin consultar los datos de prueba final [13]. Repetir entrenamientos.\nSi cambia la tarea del artículo, ejecutar también su modelo en nuestra tarea.'
}}
with ZipFile(PPT) as z:
    infos = z.infolist()
    before = {i.filename: z.read(i.filename) for i in infos}
after = dict(before)
for slide, edits in changes.items():
    key = f'ppt/slides/slide{slide}.xml'
    root = E.fromstring(before[key])
    for shape in root.findall('.//p:sp', NS):
        ident = int(shape.find('.//p:cNvPr', NS).get('id'))
        if ident not in edits:
            continue
        body = shape.find('p:txBody', NS)
        old = body.findall('a:p', NS)
        template = old[0]
        rpr = template.find('.//a:rPr', NS)
        for p in old:
            body.remove(p)
        for line in edits[ident].split('\n'):
            p = E.SubElement(body, '{'+NS['a']+'}p')
            pp = template.find('a:pPr', NS)
            if pp is not None:
                p.append(deepcopy(pp))
            r = E.SubElement(p, '{'+NS['a']+'}r')
            if rpr is not None:
                props = deepcopy(rpr)
                if ident in [235,238,241,244,266,268]:
                    props.set('sz', '1800')
                if ident in [266,268]:
                    props.set('sz', '1650')
                if ident == 267:
                    props.set('sz', '2000')
                r.append(props)
            E.SubElement(r, '{'+NS['a']+'}t').text = line
    after[key] = E.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
changed = [k for k in before if before[k] != after[k]]
assert set(changed) == {'ppt/slides/slide13.xml', 'ppt/slides/slide14.xml'}
out = HERE / PPT.name
with ZipFile(out, 'w') as z:
    for info in infos:
        z.writestr(info, after[info.filename])
(HERE / 'CLARITY_VERIFICATION.json').write_text(json.dumps({
    'input_sha256': hashlib.sha256(PPT.read_bytes()).hexdigest(),
    'output_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
    'changed_parts': changed, 'other_parts_byte_identical': True,
    'text_changes': changes}, ensure_ascii=False, indent=2))
print(out)
