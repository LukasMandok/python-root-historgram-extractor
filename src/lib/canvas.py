"""Optional extraction of histograms stored inside ROOT canvases."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Tuple, Union


PathLike = Union[str, Path]
CanvasHistogram = Tuple[str, str, int, int]


def find_canvas_histograms(root_path: PathLike) -> List[CanvasHistogram]:
    """Return searchable references to TH1 objects stored in TCanvas pads."""
    source_path = Path(root_path).expanduser().resolve()
    root_command = shutil.which("root")
    if root_command is None:
        return []

    source_literal = json.dumps(str(source_path))
    expression = (
        f'TFile *input=TFile::Open({source_literal}); '
        f'if(!input||input->IsZombie()) gSystem->Exit(2); '
        f'TIter canvas_keys(input->GetListOfKeys()); TKey *canvas_key; '
        f'while((canvas_key=(TKey*)canvas_keys())) {{ '
        f'if(!TString(canvas_key->GetClassName()).BeginsWith("TCanvas")) continue; '
        f'TCanvas *canvas=(TCanvas*)input->Get(canvas_key->GetName()); '
        f'if(!canvas) continue; '
        f'TIter pads(canvas->GetListOfPrimitives()); TObject *pad_object; int pad_index=0; '
        f'while((pad_object=pads())) {{ TPad *pad=(TPad*)pad_object; '
        f'if(!pad->InheritsFrom("TPad")) {{ ++pad_index; continue; }} '
        f'TIter objects(pad->GetListOfPrimitives()); TObject *object; int histogram_index=0; '
        f'while((object=objects())) {{ if(object->InheritsFrom("TH1")) {{ '
        f'std::cout << "__CANVAS_HIST__\\t" << canvas_key->GetName() << "\\t" '
        f'<< canvas->GetListOfPrimitives()->IndexOf(pad) << "\\t" << histogram_index '
        f'<< "\\t" << object->GetName() << "\\t" << object->GetTitle() << std::endl; '
        f'++histogram_index; }} }} ++pad_index; }} }} input->Close();'
    )
    result = subprocess.run(
        [root_command, "-l", "-b", "-q", "-e", expression],
        capture_output=True,
        text=True,
        check=False,
    )
    references: List[CanvasHistogram] = []
    for line in result.stdout.splitlines():
        if not line.startswith("__CANVAS_HIST__\t"):
            continue
        fields = line.split("\t", 5)
        if len(fields) == 6:
            _, canvas, pad, histogram, name, title = fields
            references.append((
                f"canvas://{canvas}|pad={pad}|hist={histogram}|name={name}|title={title}",
                canvas,
                int(pad),
                int(histogram),
            ))
    return references


def select_canvas_histograms(
    references: List[CanvasHistogram],
    *,
    stack: bool,
) -> List[str]:
    """Apply the same single-selection behavior as ordinary ROOT keys."""
    if not references:
        return []
    if stack or len(references) == 1:
        return [reference[0] for reference in references]

    print("\nFound multiple histograms in canvases:")
    for index, (reference, _canvas, _pad, _histogram) in enumerate(references):
        title = reference.split("|title=", 1)[-1]
        print(f"  ({index}) {title}")

    while True:
        choice = input(
            f"\nSelect index (0-{len(references) - 1}), or ENTER for first: "
        ).strip()
        if not choice:
            return [references[0][0]]
        try:
            index = int(choice)
            if 0 <= index < len(references):
                return [references[index][0]]
        except ValueError:
            pass
        print("Invalid selection. Please try again.")


def parse_canvas_reference(reference: str) -> Tuple[str, int, int]:
    """Decode a virtual canvas histogram key."""
    prefix, metadata = reference.split("|", 1)
    canvas = prefix.removeprefix("canvas://")
    fields = dict(item.split("=", 1) for item in metadata.split("|"))
    return canvas, int(fields["pad"]), int(fields["hist"])


def is_canvas_reference(value: str) -> bool:
    return value.startswith("canvas://")


def extract_canvas_histogram(
    root_path: PathLike,
    canvas_key: str,
    *,
    pad_index: int = 0,
    histogram_index: int = 0,
) -> Path:
    """Copy a histogram from a TCanvas into a temporary ROOT file.

    ROOT is an optional external dependency used only for files containing
    canvases. Histograms are selected among drawable TH1-derived objects in
    the requested pad, in their stored order.
    """
    source_path = Path(root_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"ROOT file not found: {source_path}")
    if pad_index < 0 or histogram_index < 0:
        raise ValueError("pad_index and histogram_index must be non-negative")

    root_command = shutil.which("root")
    if root_command is None:
        raise RuntimeError(
            "Reading TCanvas objects requires the CERN ROOT executable on PATH."
        )

    output_file = NamedTemporaryFile(suffix=".root", delete=False)
    output_file.close()
    output_path = Path(output_file.name)

    source_literal = json.dumps(str(source_path))
    output_literal = json.dumps(str(output_path))
    canvas_literal = json.dumps(canvas_key)
    expression = (
        f'TFile *input=TFile::Open({source_literal}); '
        f'if(!input||input->IsZombie()) gSystem->Exit(2); '
        f'TCanvas *canvas=(TCanvas*)input->Get({canvas_literal}); '
        f'if(!canvas) gSystem->Exit(3); '
        f'TPad *pad=(TPad*)canvas->GetListOfPrimitives()->At({pad_index}); '
        f'if(!pad) gSystem->Exit(4); '
        f'TObject *selected=nullptr; int found=0; '
        f'TIter iterator(pad->GetListOfPrimitives()); TObject *object; '
        f'while((object=iterator())) {{ '
        f'if(object->InheritsFrom("TH1") && found++=={histogram_index}) {{ selected=object; break; }} '
        f'}} '
        f'if(!selected) gSystem->Exit(5); '
        f'TFile *output=TFile::Open({output_literal},"RECREATE"); '
        f'selected->Write("selected_histogram"); output->Close(); input->Close();'
    )

    result = subprocess.run(
        [root_command, "-l", "-b", "-q", "-e", expression],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not output_path.stat().st_size:
        output_path.unlink(missing_ok=True)
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(
            f"Could not extract histogram {histogram_index} from canvas "
            f"'{canvas_key}' pad {pad_index}. {details}"
        )

    return output_path