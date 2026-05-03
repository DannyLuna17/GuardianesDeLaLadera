from __future__ import annotations


def build_driver_chips(drivers: dict) -> list[str]:
    chips: list[str] = []
    rain_72h = drivers.get("rain_72h")
    rain_24h = drivers.get("rain_24h")
    slope_deg = drivers.get("slope_deg")
    geology_class = drivers.get("geology_class")
    soil_class = drivers.get("soil_class")
    if rain_72h is not None and rain_72h >= 140:
        chips.append(f"Acumulado 72h {drivers['rain_72h']} mm")
    if rain_24h is not None and rain_24h >= 80:
        chips.append(f"Lluvia 24h {drivers['rain_24h']} mm")
    if slope_deg is not None and slope_deg >= 25:
        chips.append(f"Pendiente > {drivers['slope_deg']} deg")
    if drivers.get("deforestation_proxy") is not None and drivers["deforestation_proxy"] >= 0.4:
        chips.append(f"Perdida cobertura {round(drivers['deforestation_proxy'] * 100)}%")
    if geology_class:
        chips.append(f"Geologia {geology_class}")
    if soil_class:
        chips.append(f"Suelo {soil_class}")
    return chips


def build_suggestions(
    zone_name: str,
    municipality_name: str,
    risk_level: str,
    road_segment_names: list[str],
) -> list[str]:
    suggestions = [
        f"Priorizar revision preventiva en {zone_name} ({municipality_name}).",
        "Verificar drenajes, taludes y puntos con saturacion reciente.",
    ]
    if road_segment_names:
        suggestions.append(
            "Monitorear segmentos viales cercanos: " + ", ".join(sorted(road_segment_names[:3])) + "."
        )
    if risk_level in {"Naranja", "Rojo"}:
        suggestions.append("Mantener seguimiento reforzado durante la siguiente ventana operativa.")
    return suggestions


def build_summary(
    zone_name: str,
    municipality_name: str,
    risk_text: str,
    drivers: dict,
    event_count: int,
) -> str:
    rain_72h = drivers.get("rain_72h")
    rain_24h = drivers.get("rain_24h")
    slope_deg = drivers.get("slope_deg")
    rain_clause = (
        f"acumulados de lluvia de {rain_72h} mm en 72h y {rain_24h} mm en 24h"
        if rain_72h is not None and rain_24h is not None
        else "las ultimas observaciones oficiales disponibles"
    )
    terrain_clause = (
        f"pendientes de {slope_deg} deg"
        if slope_deg is not None
        else "la configuracion espacial y los antecedentes historicos disponibles"
    )
    return (
        f"El sistema indica riesgo {risk_text} en {zone_name} ({municipality_name}). "
        f"La lectura combina {rain_clause}, {terrain_clause} y {event_count} antecedentes historicos en el municipio."
    )
