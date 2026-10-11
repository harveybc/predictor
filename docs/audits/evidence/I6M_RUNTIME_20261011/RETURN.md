# Ejecutor multihorizonte: verificacion de software

El test se escribio primero: coleccion PRE rechazo la ausencia de i6m_data.
POST: 36 pruebas pasan (contratos y TensorFlow real), CPU, bajo crispdm-run,
3 GiB y 180 s. La suite completa heredada no se ejecuto.

Se probaron ventanas 3,5,7,9,13,49,241: 34921 parametros; mascara de mercado,
perturbacion causal del prefijo, guardado/carga exactos. El factory nuevo
reproduce pesos y forward bit a bit del sellado en 3,5,9. ARIMA dinamico no
condiciona en observaciones futuras. Cola exclusiva, identidad/lease, reinicio,
barrera de etapas, purga por h, precios float64 y signo de skill probados.

Poblacion de VALIDATION 2024 autenticada por horizonte (horas: origenes):
2:6130, 3:6076, 4:6023, 5:5969, 6:5915, 24:4953, 48:3685,
72:3712, 96:3712, 120:3683. Se conserva el mismo vector dentro de cada h.
No TEST, entrenamiento nuevo ni exactitud financiera medidos por estas pruebas.

MH01-MH06 cubiertos por pruebas de contrato/modelo; MH07 cola/restart unitario
y MH08 esquema/outbox necesitan observacion operativa; MH09 pilotos reales y
MH10 cierre total se registran en el directorio privado de campana. No se
declaran completos anticipadamente. No se editaron motores sellados.
