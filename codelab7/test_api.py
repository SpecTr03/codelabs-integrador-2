# Cliente de ejemplo para probar la API
# Requisitos: pip install requests

import requests
import json

# Configuración
API_URL = "http://localhost:8000"

def test_health():
    """Verificar estado de la API"""
    print("\n🔍 Verificando estado de la API...")
    response = requests.get(f"{API_URL}/health")
    print(f"Estado: {response.status_code}")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))

def test_single_prediction(texto):
    """Probar predicción individual"""
    print(f"\n📝 Probando predicción individual...")
    print(f"Texto: '{texto}'")
    
    response = requests.post(
        f"{API_URL}/predict",
        json={"texto": texto}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Resultado:")
        print(f"   Predicción: {result['prediccion'].upper()}")
        print(f"   Confianza: {result['confianza']:.4f}" if result['confianza'] else "   Confianza: N/A")
        print(f"   Texto limpio: '{result['texto_limpio']}'")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())

def test_batch_prediction(textos):
    """Probar predicción en lote"""
    print(f"\n📋 Probando predicción en lote ({len(textos)} comentarios)...")
    
    response = requests.post(
        f"{API_URL}/predict/batch",
        json={"textos": textos}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Resultados:")
        print(f"   Total procesados: {result['total']}")
        print(f"   Positivos: {result['resumen']['positivos']} ({result['resumen']['porcentaje_positivos']}%)")
        print(f"   Negativos: {result['resumen']['negativos']} ({result['resumen']['porcentaje_negativos']}%)")
        
        print("\n   Detalle por comentario:")
        for i, pred in enumerate(result['predicciones'], 1):
            emoji = "😊" if pred['prediccion'] == "positivo" else "😡"
            conf_str = f" (conf: {pred['confianza']:.2f})" if pred['confianza'] else ""
            print(f"   {i}. {emoji} {pred['prediccion'].upper()}{conf_str}")
            print(f"      '{pred['texto_original'][:60]}...'")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.json())

if __name__ == "__main__":
    print("="*70)
    print("🧪 PRUEBAS DEL CLIENTE API - CLASIFICADOR DE COMENTARIOS")
    print("="*70)
    
    # 1. Verificar estado
    test_health()
    
    # 2. Probar predicción individual
    test_single_prediction("Excelente servicio, muy recomendable 😊")
    test_single_prediction("Pésima experiencia, no lo recomiendo para nada")
    test_single_prediction("El producto llegó en perfectas condiciones, 10/10")
    
    # 3. Probar predicción en lote
    comentarios_prueba = [
        "El envío fue rapidísimo y el empaque llegó impecable, gracias!",
        "Demoraron demasiado y además nadie respondió los mensajes",
        "Calidad/precio brutal, quedé muy satisfecho",
        "No lo recomiendo, salió defectuoso y me tocó devolverlo",
        "Súper contento con la compra, volveré a comprar",
        "Producto de mala calidad, se rompió al primer uso",
        "Atención al cliente excepcional, resolvieron todo rápido",
        "Nunca llegó el pedido y no me responden los emails"
    ]
    test_batch_prediction(comentarios_prueba)
    
    print("\n" + "="*70)
    print("✅ Pruebas completadas")
    print("="*70)
