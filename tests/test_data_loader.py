from pathlib import Path
from src.data_loader import get_files

def test_get_files_returns_only_files_sorted(tmp_path: Path):
    # Arrange: lav en mappe med filer og en undermappe
    (tmp_path / "b.txt").write_text("b")
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "c.txt").write_text("c")

    # Act
    files = get_files(tmp_path)

    # Assert: kun filer i tmp_path (ikke mapper), og sorteret
    assert all(p.is_file() for p in files)
    assert [p.name for p in files] == ["a.txt", "b.txt"]