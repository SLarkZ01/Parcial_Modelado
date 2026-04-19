#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def flatten_dict(data, parent_key="", sep="."):
    out = {}
    if not isinstance(data, dict):
        return out

    for key, value in data.items():
        full_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            out.update(flatten_dict(value, full_key, sep=sep))
        elif isinstance(value, list):
            out[full_key] = json.dumps(value, ensure_ascii=False)
        else:
            out[full_key] = value
    return out


def detect_format(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                break
            if ch.isspace():
                continue
            if ch == "[":
                return "array"
            if ch == "{":
                return "jsonl"
            break
    raise ValueError("No se pudo detectar el formato JSON (array/jsonl).")


def iter_jsonl_records(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Linea {line_no} invalida en JSONL: {exc}") from exc
            if isinstance(obj, dict):
                yield obj


def iter_json_array_records(file_path, chunk_size=1024 * 1024):
    decoder = json.JSONDecoder()
    buf = ""
    idx = 0
    eof = False
    state = "start"

    with open(file_path, "r", encoding="utf-8") as f:
        def read_more():
            nonlocal buf, eof
            if eof:
                return False
            chunk = f.read(chunk_size)
            if chunk:
                buf += chunk
                return True
            eof = True
            return False

        while True:
            if idx >= len(buf) and not eof:
                read_more()

            if idx > 0 and idx > len(buf) // 2:
                buf = buf[idx:]
                idx = 0

            while idx < len(buf) and buf[idx].isspace():
                idx += 1

            if state == "start":
                if idx >= len(buf):
                    if eof:
                        break
                    continue
                if buf[idx] != "[":
                    raise ValueError("El archivo no es un JSON array valido (esperaba '[').")
                idx += 1
                state = "before_item"
                continue

            if state == "before_item":
                while idx < len(buf) and buf[idx].isspace():
                    idx += 1

                if idx >= len(buf):
                    if eof:
                        raise ValueError("JSON incompleto al buscar el siguiente item.")
                    continue

                if buf[idx] == "]":
                    return

                if buf[idx] == ",":
                    idx += 1
                    continue

                try:
                    obj, end = decoder.raw_decode(buf, idx)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError("JSON invalido o incompleto en el array.")
                    read_more()
                    continue

                idx = end
                if isinstance(obj, dict):
                    yield obj
                state = "after_item"
                continue

            if state == "after_item":
                while idx < len(buf) and buf[idx].isspace():
                    idx += 1

                if idx >= len(buf):
                    if eof:
                        raise ValueError("JSON incompleto despues de un item.")
                    continue

                if buf[idx] == ",":
                    idx += 1
                    state = "before_item"
                    continue
                if buf[idx] == "]":
                    return

                raise ValueError("JSON array invalido: esperaba ',' o ']'.")


def get_record_iterator(file_path, fmt):
    if fmt == "jsonl":
        return iter_jsonl_records(file_path)
    return iter_json_array_records(file_path)


def collect_headers(file_path, fmt, limit=0):
    headers = set()
    count = 0
    for record in get_record_iterator(file_path, fmt):
        flat = flatten_dict(record)
        headers.update(flat.keys())
        count += 1
        if limit > 0 and count >= limit:
            break
    return sorted(headers), count


def write_csv(file_path, output_path, fmt, headers):
    rows = 0
    with open(output_path, "w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()

        for record in get_record_iterator(file_path, fmt):
            flat = flatten_dict(record)
            writer.writerow({k: flat.get(k, "") for k in headers})
            rows += 1
    return rows


def main():
    parser = argparse.ArgumentParser(description="Convierte JSON grande a CSV en streaming.")
    parser.add_argument("input_json", help="Archivo JSON de entrada")
    parser.add_argument("output_csv", help="Archivo CSV de salida")
    parser.add_argument("--format", choices=["auto", "array", "jsonl"], default="auto")
    parser.add_argument(
        "--header-scan-limit",
        type=int,
        default=0,
        help="Cantidad de registros para descubrir columnas (0 = escaneo completo)",
    )

    args = parser.parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_csv)

    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    fmt = detect_format(input_path) if args.format == "auto" else args.format

    headers, scanned = collect_headers(input_path, fmt, limit=args.header_scan_limit)
    if not headers:
        raise ValueError("No se detectaron columnas en el JSON.")

    total_rows = write_csv(input_path, output_path, fmt, headers)

    print(f"OK: CSV generado en {output_path}")
    print(f"Formato usado: {fmt}")
    print(f"Columnas: {len(headers)}")
    print(f"Filas escritas: {total_rows}")
    if args.header_scan_limit > 0:
        print(
            "Aviso: se usaron columnas detectadas en "
            f"{scanned} registros para definir el header."
        )


if __name__ == "__main__":
    main()
