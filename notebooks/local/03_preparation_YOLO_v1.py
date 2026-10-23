from pathlib import Path
from collections import defaultdict
import random
import shutil
import hashlib

# Exportaciones originales de CVAT
ROOT = Path(__file__).resolve().parent

TASKS = {
    "wasps": ROOT / "data/processed/cvat/task_wasps",
    "varroa": ROOT / "data/processed/cvat/task_varroa",
}

OUTPUT = ROOT / "data/processed/yolo_v1"

CLASS_NAMES = {
    0: "bee",
    1: "varroa",
    2: "wasp",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
VAL_RATIO = 0.20


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_label_file(path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Formato inválido: {path.name}: {line}")

        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]

        if class_id not in CLASS_NAMES:
            raise ValueError(f"Clase desconocida en {path.name}: {class_id}")

        if not all(0 <= value <= 1 for value in coords):
            raise ValueError(f"Coordenadas fuera de rango: {path.name}")

        if coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f"Ancho/alto inválido: {path.name}")

    return lines


def collect_task(task_name, task_root):
    images_dir = task_root / "images"
    labels_dir = task_root / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"No se encontraron images/ y labels/ en {task_root}"
        )

    image_paths = [
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    labels_by_stem = {p.stem: p for p in labels_dir.rglob("*.txt")}
    annotated = []
    pending = []

    for image in image_paths:
        label_path = labels_by_stem.get(image.stem)

        if label_path is None:
            pending.append(image)
            continue

        lines = read_label_file(label_path)

        if not lines:
            # No asumimos que una etiqueta vacía sea una imagen negativa.
            pending.append(image)
            continue

        annotated.append((image, lines, task_name))

    return annotated, pending


def reset_output():
    # Solo elimina la carpeta de salida de este experimento.
    # No modifica las exportaciones originales de CVAT.
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    for split in ("train", "val"):
        (OUTPUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    (OUTPUT / "images" / "predict").mkdir(parents=True, exist_ok=True)


def main():
    all_annotated = []
    all_pending = []
    seen_hashes = {}

    for task_name, task_root in TASKS.items():
        annotated, pending = collect_task(task_name, task_root)

        print(
            f"{task_name}: "
            f"{len(annotated)} imágenes anotadas; "
            f"{len(pending)} pendientes"
        )

        all_annotated.extend(annotated)
        all_pending.extend((image, task_name) for image in pending)

    # Detectar imágenes idénticas por contenido.
    unique_annotated = []
    duplicate_count = 0

    for image, lines, task_name in all_annotated:
        digest = sha256(image)

        if digest in seen_hashes:
            duplicate_count += 1
            print(
                f"Duplicado anotado: {image.name} "
                f"(ya existe como {seen_hashes[digest]})"
            )
            continue

        seen_hashes[digest] = image.name
        unique_annotated.append((image, lines, task_name))

    # División aleatoria por imagen.
    random.Random(SEED).shuffle(unique_annotated)
    val_count = max(1, round(len(unique_annotated) * VAL_RATIO))
    val_items = unique_annotated[:val_count]
    train_items = unique_annotated[val_count:]

    reset_output()

    # Copiar imágenes y etiquetas anotadas.
    used_names = set()

    def copy_annotated(items, split):
        for image, lines, task_name in items:
            # Evitar colisiones de nombres entre tareas.
            filename = f"{task_name}__{image.name}"
            if filename in used_names:
                raise ValueError(f"Nombre duplicado: {filename}")
            used_names.add(filename)

            shutil.copy2(image, OUTPUT / "images" / split / filename)

            label_name = Path(filename).with_suffix(".txt")
            label_path = OUTPUT / "labels" / split / label_name
            label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    copy_annotated(train_items, "train")
    copy_annotated(val_items, "val")

    # Copiar imágenes pendientes para preanotación.
    pending_hashes = set()
    predict_count = 0

    for image, task_name in all_pending:
        digest = sha256(image)

        # Si es una imagen ya anotada, no volver a enviarla a predicción.
        if digest in seen_hashes:
            continue

        if digest in pending_hashes:
            continue

        pending_hashes.add(digest)
        filename = f"{task_name}__{image.name}"
        shutil.copy2(image, OUTPUT / "images" / "predict" / filename)
        predict_count += 1

    # Archivo de configuración YOLO.
    yaml_text = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
        "  0: bee\n"
        "  1: varroa\n"
        "  2: wasp\n"
    )
    (OUTPUT / "data.yaml").write_text(yaml_text, encoding="utf-8")

    # Informe.
    report = [
        "AUDITORÍA DATASET YOLO V1",
        f"Imágenes anotadas únicas: {len(unique_annotated)}",
        f"Entrenamiento: {len(train_items)}",
        f"Validación: {len(val_items)}",
        f"Duplicados anotados omitidos: {duplicate_count}",
        f"Imágenes pendientes copiadas para predicción: {predict_count}",
        "",
        "Clases:",
        "0 = bee",
        "1 = varroa",
        "2 = wasp",
        "",
        "Nota: las predicciones deberán revisarse y corregirse manualmente.",
    ]

    (OUTPUT / "audit_report.txt").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print("\n".join(report))
    print(f"\nDataset creado en: {OUTPUT}")


if __name__ == "__main__":
    main()
