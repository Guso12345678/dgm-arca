import argparse
import json
import random

#Intrucción de formato para el verifier y para todas las partes porque sino va a fallar nuestro primer agente (esta es la versión basica)
INSTRUCCION_REDONDEO = (
    "Da la respuesta final redondeada a 2 decimales, dentro de <answer></answer>, "
    "sin unidades ni símbolos adicionales (por ejemplo: 345.67, no 345.67€ ni ~345.7)."
)

def construir_enunciado(texto_base: str) -> str:
    return f"{texto_base}\n\n{INSTRUCCION_REDONDEO}"



#Formulas cerradas: 
def cuota_francesa(principal: float, tipo_nominal_anual: float, meses: int) -> float:
    i = (tipo_nominal_anual / 100) / 12
    return principal * i / (1 - (1 + i) ** (-meses))
 
 
def ratio_endeudamiento(cuota: float, otras_deudas: float, ingresos_netos: float) -> float:
    return (cuota + otras_deudas) / ingresos_netos * 100
 
 
def coste_total(cuota: float, meses: int, comision_apertura: float, principal: float) -> float:
    return cuota * meses + comision_apertura - principal

#Tae: 
def _valor_actual_cuotas(cuota: float, meses: int, i: float) -> float:
    return sum(cuota / (1 + i) ** k for k in range(1, meses + 1))
 
 
def resolver_tae(principal: float, comision_apertura: float, cuota: float, meses: int,
                  tol: float = 1e-10, max_iter: int = 200) -> float:
    p_neto = principal - comision_apertura
    lo, hi = 0.0, 2.0  # tipo mensual entre 0% y 200% -- margen de sobra
 
    for _ in range(max_iter):
        mid = (lo + hi) / 2
        f_mid = _valor_actual_cuotas(cuota, meses, mid) - p_neto
        if abs(f_mid) < tol:
            break
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    i_tae_mensual = (lo + hi) / 2
    return ((1 + i_tae_mensual) ** 12 - 1) * 100


#Generacion de prestamos sinteticos: 

PLAZOS = [12, 24, 36, 48, 60]
 
 
def generar_prestamo(rng: random.Random) -> dict:
    principal = round(rng.uniform(1000, 30000), 2)
    meses = rng.choice(PLAZOS)
    tipo_nominal = round(rng.uniform(5.0, 15.0), 2)
    comision_apertura = round(principal * rng.uniform(0.0, 0.02), 2)  # 0-2% del principal
    ingresos_netos = round(rng.uniform(900, 4000), 2)
    otras_deudas = round(rng.uniform(0, 500), 2)
    return dict(
        principal=principal, meses=meses, tipo_nominal=tipo_nominal,
        comision_apertura=comision_apertura, ingresos_netos=ingresos_netos,
        otras_deudas=otras_deudas,
    )
 
 
def texto_prestamo(p: dict) -> str:
    return (
        f"Una persona con ingresos netos de {p['ingresos_netos']:.2f} euros/mes "
        f"(y otras deudas mensuales de {p['otras_deudas']:.2f} euros) solicita un "
        f"préstamo de {p['principal']:.2f} euros a {p['meses']} meses con un tipo "
        f"nominal anual del {p['tipo_nominal']:.2f}% y una comisión de apertura de "
        f"{p['comision_apertura']:.2f} euros."
    )
 
 
def generar_caso(idx: int, tipo: str, rng: random.Random) -> dict:
    p = generar_prestamo(rng)
    cuota = cuota_francesa(p["principal"], p["tipo_nominal"], p["meses"])
    cuota = round(cuota, 2)
    base = texto_prestamo(p)
 
    if tipo == "cuota":
        pregunta = f"{base} Calcula la cuota mensual mediante el sistema de amortización francés."
        respuesta = cuota
    elif tipo == "ratio_endeudamiento":
        pregunta = (
            f"{base} La cuota mensual de este préstamo es de {cuota:.2f} euros. "
            f"Calcula el ratio de endeudamiento resultante (en %), incluyendo esta "
            f"cuota y las otras deudas mensuales ya existentes."
        )
        respuesta = ratio_endeudamiento(cuota, p["otras_deudas"], p["ingresos_netos"])
    elif tipo == "coste_total":
        pregunta = (
            f"{base} La cuota mensual de este préstamo es de {cuota:.2f} euros. "
            f"Calcula el coste total del crédito en euros (la diferencia entre "
            f"todo lo que se paga en total y el capital recibido)."
        )
        respuesta = coste_total(cuota, p["meses"], p["comision_apertura"], p["principal"])
    elif tipo == "tae":
        pregunta = (
            f"{base} La cuota mensual de este préstamo es de {cuota:.2f} euros. "
            f"Calcula la Tasa Anual Equivalente (TAE) de este préstamo, en %."
        )
        respuesta = resolver_tae(p["principal"], p["comision_apertura"], cuota, p["meses"])
    else:
        raise ValueError(tipo)
 
    return {
        "id": f"{tipo}_{idx:05d}",
        "tipo": tipo,
        "question": construir_enunciado(pregunta),
        "answer": f"{round(respuesta, 2):.2f}",
        "_debug_prestamo": p,
        "_debug_cuota": round(cuota, 2),
    }
 
 
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_por_tipo", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frac_test", type=float, default=0.15)
    ap.add_argument("--out_train", type=str, default="rlm_data_train.jsonl")
    ap.add_argument("--out_test", type=str, default="rlm_data_test.jsonl")
    args = ap.parse_args()
 
    rng = random.Random(args.seed)
    tipos = ["cuota", "ratio_endeudamiento", "coste_total", "tae"]
 
    registros = []
    for tipo in tipos:
        for i in range(args.n_por_tipo):
            registros.append(generar_caso(i, tipo, rng))
    rng.shuffle(registros)
 
    n_test = int(len(registros) * args.frac_test)
    test, train = registros[:n_test], registros[n_test:]
 
    for path, rows in [(args.out_train, train), (args.out_test, test)]:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{len(rows)} problemas -> {path}")
 
    print("Ejemplo de cada tipo:")
    for tipo in tipos:
        ejemplo = next(r for r in registros if r["tipo"] == tipo)
        print(f"--- {tipo} ---")
        print(json.dumps({k: v for k, v in ejemplo.items() if not k.startswith("_debug")},
                          ensure_ascii=False, indent=2))
        print()
 
 
if __name__ == "__main__":
    main()