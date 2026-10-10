"""Saved source is evidence, not executable pytest input."""
collect_ignore = ['artifacts']


def pytest_ignore_collect(collection_path, config):
    # Alternate --output directories under E1 are also evidence. The runner
    # creates this marker before writing any source snapshot.
    if collection_path.is_dir() and (collection_path / 'run_status.json').is_file():
        return True
    return None
