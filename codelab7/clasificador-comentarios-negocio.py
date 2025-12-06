# Requisitos (si falta): pip install scikit-learn pandas numpy joblib matplotlib

import re, random, numpy as np, pandas as pd, unicodedata
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
import pandas as pd
import joblib

# ----------------------------
# 1) Dataset sintético realista
# ----------------------------
random.seed(42); np.random.seed(42) # Explicacion detallada: Esto asegura que los resultados sean reproducibles al fijar la semilla para las funciones aleatorias de Python y NumPy.

positivos = [
    "Excelente servicio 😊","Muy buena atención","Me encantó el producto ❤️",
    "Rápido y confiable","Todo llegó perfecto 👌","Calidad superior",
    "Lo recomiendo totalmente 💯","Volveré a comprar","Precio justo y buena calidad",
    "El soporte fue amable","Experiencia increíble 🎉","Funcionó mejor de lo esperado",
    "Entregado a tiempo ⏰","Muy satisfecho","Cinco estrellas ⭐⭐⭐⭐⭐",
    "La comida estaba deliciosa 🍕","El empaque impecable","Súper recomendable",
    "Buen trato del personal","Gran experiencia",
    "Increíble atención al cliente, resolvieron mi problema en minutos",
    "El producto superó mis expectativas, vale cada peso",
    "Envío rapidísimo, llegó en 24 horas 🚚",
    "Calidad premium, se nota la diferencia con otras marcas",
    "Excelente relación calidad-precio, totalmente satisfecho",
    "El mejor lugar para comprar, siempre cumplen",
    "Producto original y bien empaquetado, gracias!",
    "Servicio de 10, muy profesionales y atentos",
    "Quedé fascinado con la compra, volveré sin dudarlo",
    "Todo perfecto desde el pedido hasta la entrega",
    "Recomiendo 100%, no tengo quejas",
    "La atención fue excepcional, muy amables",
    "Producto tal cual se describe, muy conforme",
    "Envío seguro y rápido, excelente experiencia",
    "Me salvaron con el pedido urgente, gracias totales",
    "Calidad excepcional, definitivamente volveré",
    "Súper contento con la compra 😄",
    "El mejor servicio que he recibido en años",
    "Todo llegó en perfectas condiciones, muy bien empaquetado",
    "Atención personalizada y eficiente, felicitaciones",
    "Producto de primera calidad, lo recomiendo ampliamente"
]

negativos = [
    "Pésimo servicio 😡","Muy mala atención","Odio este producto 👎",
    "Lento y poco confiable","Llegó dañado 💔","Calidad terrible",
    "No lo recomiendo ❌","No vuelvo a comprar","Caro y mala calidad",
    "El soporte fue grosero","Experiencia horrible 😠","Peor de lo esperado",
    "Entregado tarde ⏱️","Muy decepcionado","Una estrella ⭐",
    "La comida estaba fría 🥶","El empaque roto","Nada recomendable",
    "Mal trato del personal","Mala experiencia",
    "Nunca llegó el pedido y nadie responde los mensajes",
    "Producto defectuoso, solicité devolución y me ignoraron",
    "Tardaron 3 semanas en entregar algo que prometieron en 3 días",
    "Calidad pésima, se rompió al segundo día de uso",
    "No coincide con la descripción, me siento estafado",
    "Servicio al cliente horrible, nadie resuelve nada",
    "Llegó todo aplastado y maltratado",
    "Carísimo para la basura que envían",
    "Decepción total, no vuelvo a comprar aquí jamás",
    "El producto es una porquería, no funciona",
    "Pésima experiencia, perdí mi dinero",
    "Nadie responde los reclamos, irresponsables",
    "Material de mala calidad, se daña fácilmente",
    "Envío demorado y producto en mal estado",
    "No recomiendo para nada, muy mala experiencia",
    "Atención al cliente inexistente, no resuelven nada",
    "Fraude total, no es lo que ofrecen",
    "Peor compra de mi vida, no caigan en esto",
    "Producto falso y de ínfima calidad",
    "Decepcionante en todos los aspectos, eviten comprar aquí",
    "No vale la pena, hay mejores opciones"
]

def variantes(frase):
    extras = [
        "", "!", "!!", "!!!", " 🙂", " 😡", " 😍", " 👍", " 👎",
        " de verdad", " en serio", " 10/10", " 1/10", " 0/10",
        " súper", " la verdad", " jamás", " nunca", " para nada",
        " 100% recomendado", " no lo compren", " GRACIAS", " HORRIBLE",
        " ver más en https://ejemplo.com", " visita www.tienda.com",
        " @usuario gracias", " contacto@correo.com", " #recomendado", " #fraude",
        "...", " sin palabras", " increíble", " fatal"
    ]
    return frase + random.choice(extras)

pos = [variantes(p) for _ in range(6) for p in positivos] # Aumentar datos con variantes
neg = [variantes(n) for _ in range(6) for n in negativos] # Aumentar datos con variantes
textos = pos + neg # Mezclar positivos y negativos
etiquetas = [1]*len(pos) + [0]*len(neg) # 1=Positivo, 0=Negativo

df = pd.DataFrame({"texto": textos, "etiqueta": etiquetas}).sample(frac=1, random_state=42).reset_index(drop=True)
# Mezclar filas aleatoriamente

print("Muestras:", df.shape[0], " | Positivos:", df.etiqueta.sum(), " | Negativos:", len(df)-df.etiqueta.sum())

# ----------------------------
# 2) Limpieza avanzada
# ----------------------------
def limpiar(s: str) -> str:
    """Pipeline de limpieza avanzado:
    1. Convertir a minúsculas
    2. Eliminar URLs
    3. Eliminar emails
    4. Eliminar menciones (@usuario)
    5. Eliminar hashtags (conservar texto)
    6. Eliminar emojis
    7. Normalizar acentos (opcional, aquí los conservamos)
    8. Eliminar caracteres especiales
    9. Normalizar espacios
    """
    # 1. Minúsculas
    s = s.lower()
    
    # 2. Eliminar URLs (http, https, www)
    s = re.sub(r'https?://\S+|www\.\S+', ' ', s)
    
    # 3. Eliminar emails
    s = re.sub(r'\S+@\S+', ' ', s)
    
    # 4. Eliminar menciones (@usuario)
    s = re.sub(r'@\w+', ' ', s)
    
    # 5. Eliminar hashtags pero conservar el texto (#recomendado -> recomendado)
    s = re.sub(r'#(\w+)', r'\1', s)
    
    # 6. Eliminar emojis (rangos Unicode principales)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticonos
        "\U0001F300-\U0001F5FF"  # símbolos y pictogramas
        "\U0001F680-\U0001F6FF"  # transporte y símbolos de mapas
        "\U0001F1E0-\U0001F1FF"  # banderas
        "\U00002702-\U000027B0"  # dingbats
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # símbolos y emoticonos suplementarios
        "\U00002600-\U000026FF"  # símbolos misceláneos
        "\U00002700-\U000027BF"
        "]+", flags=re.UNICODE
    )
    s = emoji_pattern.sub(' ', s)
    
    # 7. Opcional: Normalizar acentos (descomponer y quitar diacríticos)
    # s = unicodedata.normalize('NFD', s)
    # s = ''.join(char for char in s if unicodedata.category(char) != 'Mn')
    # Aquí conservamos los acentos para mejor contexto en español
    
    # 8. Conservar solo letras (con acentos), números y espacios
    s = re.sub(r"[^a-záéíóúñü0-9\s]", " ", s)
    
    # 9. Normalizar espacios múltiples
    s = re.sub(r"\s+", " ", s).strip()
    
    return s

df["texto_clean"] = df["texto"].apply(limpiar)

# ----------------------------
# 3) Split estratificado + baseline
# ----------------------------
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["texto_clean"], df["etiqueta"], test_size=0.2, random_state=42, stratify=df["etiqueta"]
) # Mantener proporción de clases en train/test

# Baseline: predecir siempre la clase mayoritaria (en este caso, positivo=1 o negativo=0)
mayoritaria = int(round(y_train.mean()))  # 0 o 1
baseline = (y_test == mayoritaria).mean() # Proporción de la clase mayoritaria en test
print(f"Baseline (clase mayoritaria): {baseline:.3f}") # Ej: 0.505 si clases balanceadas

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2)
X_train = vectorizer.fit_transform(X_train_text)  # aprende vocabulario y transforma train
X_test  = vectorizer.transform(X_test_text)       # solo transforma test (no fit)

# ----------------------------
# 4) Entrenamiento y comparación de modelos
# ----------------------------
print("\n" + "="*70)
print("COMPARACIÓN DE MODELOS")
print("="*70)

# Modelo 1: LinearSVC
print("\n[1] LinearSVC")
clf_svc = LinearSVC(class_weight="balanced", random_state=42)
clf_svc.fit(X_train, y_train)
pred_svc = clf_svc.predict(X_test)
acc_svc = accuracy_score(y_test, pred_svc)
print(f"Accuracy: {acc_svc:.4f} ({acc_svc*100:.2f}%)  |  Mejora vs baseline: {acc_svc - baseline:.4f}")
print(classification_report(y_test, pred_svc, digits=3))

# Modelo 2: LogisticRegression
print("\n[2] LogisticRegression")
clf_lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
clf_lr.fit(X_train, y_train)
pred_lr = clf_lr.predict(X_test)
acc_lr = accuracy_score(y_test, pred_lr)
print(f"Accuracy: {acc_lr:.4f} ({acc_lr*100:.2f}%)  |  Mejora vs baseline: {acc_lr - baseline:.4f}")
print(classification_report(y_test, pred_lr, digits=3))

# Comparación final
print("\n" + "="*70)
print("RESUMEN COMPARATIVO")
print("="*70)
print(f"LinearSVC:          Accuracy = {acc_svc:.4f} ({acc_svc*100:.2f}%)")
print(f"LogisticRegression: Accuracy = {acc_lr:.4f} ({acc_lr*100:.2f}%)")
mejor_modelo = "LinearSVC" if acc_svc >= acc_lr else "LogisticRegression"
print(f"\n🏆 Mejor modelo: {mejor_modelo}")
print("="*70)

# Usar el mejor modelo para el resto del código
clf = clf_svc if acc_svc >= acc_lr else clf_lr
pred = pred_svc if acc_svc >= acc_lr else pred_lr
acc = acc_svc if acc_svc >= acc_lr else acc_lr

print(f"\n\nMÉTRICAS DETALLADAS DEL MEJOR MODELO ({mejor_modelo}):")
print("="*70)

print(f"\nClase 0 (Negativo):")
print(f"  Precision: {precision_score(y_test, pred, pos_label=0):.4f} ({precision_score(y_test, pred, pos_label=0)*100:.2f}%)")
print(f"  Recall:    {recall_score(y_test, pred, pos_label=0):.4f} ({recall_score(y_test, pred, pos_label=0)*100:.2f}%)")
print(f"  F1-Score:  {f1_score(y_test, pred, pos_label=0):.4f} ({f1_score(y_test, pred, pos_label=0)*100:.2f}%)")
print(f"  Support:   {(y_test == 0).sum()}")

print(f"\nClase 1 (Positivo):")
print(f"  Precision: {precision_score(y_test, pred, pos_label=1):.4f} ({precision_score(y_test, pred, pos_label=1)*100:.2f}%)")
print(f"  Recall:    {recall_score(y_test, pred, pos_label=1):.4f} ({recall_score(y_test, pred, pos_label=1)*100:.2f}%)")
print(f"  F1-Score:  {f1_score(y_test, pred, pos_label=1):.4f} ({f1_score(y_test, pred, pos_label=1)*100:.2f}%)")
print(f"  Support:   {(y_test == 1).sum()}")

print(f"\nAccuracy Global: {acc:.4f} ({acc*100:.2f}%)")

cm = confusion_matrix(y_test, pred, labels=[0,1])
print("\nMatriz de confusión:")
print(pd.DataFrame(cm, index=["Real 0 (neg)", "Real 1 (pos)"], columns=["Pred 0 (neg)", "Pred 1 (pos)"]))

# Cross-validation para ambos modelos
print("\n" + "="*70)
print("VALIDACIÓN CRUZADA (5-fold)")
print("="*70)

pipe_svc = make_pipeline(
    TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2),
    LinearSVC(class_weight="balanced", random_state=42)
)
scores_svc = cross_val_score(pipe_svc, df["texto_clean"], df["etiqueta"], cv=5, scoring="f1_macro")
print(f"LinearSVC:          F1_macro = {scores_svc.mean():.4f} ±{scores_svc.std():.4f} ({scores_svc.mean()*100:.2f}% ±{scores_svc.std()*100:.2f}%)")

pipe_lr = make_pipeline(
    TfidfVectorizer(max_features=30000, ngram_range=(1,2), min_df=2),
    LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
)
scores_lr = cross_val_score(pipe_lr, df["texto_clean"], df["etiqueta"], cv=5, scoring="f1_macro")
print(f"LogisticRegression: F1_macro = {scores_lr.mean():.4f} ±{scores_lr.std():.4f} ({scores_lr.mean()*100:.2f}% ±{scores_lr.std()*100:.2f}%)")
print("="*70)

def predecir(textos_nuevos):
    tx = [limpiar(t) for t in textos_nuevos]
    Xn = vectorizer.transform(tx)
    p = clf.predict(Xn)
    return ["positivo" if i==1 else "negativo" for i in p]

nuevos = [
    "El envío fue rapidísimo y el empaque llegó impecable, gracias!",
    "Demoraron demasiado y además nadie respondió los mensajes",
    "Calidad/precio brutal, quedé muy satisfecho",
    "No lo recomiendo, salió defectuoso y me tocó devolverlo"
]
print("\nPredicciones en textos nuevos:")
for t, etiqueta in zip(nuevos, predecir(nuevos)):
    print(f"- {t}  ->  {etiqueta}")

joblib.dump(vectorizer, "tfidf.joblib")
joblib.dump(clf, "modelo.joblib")
print("\nModelo y vectorizador guardados.")

vec = joblib.load("tfidf.joblib")
model = joblib.load("modelo.joblib")
Xn = vec.transform(["La compra fue excelente, todo perfecto"])
print("Pred loaded model:", "positivo" if model.predict(Xn)[0]==1 else "negativo")