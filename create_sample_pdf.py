
from pathlib import Path

import pymupdf


output_path = Path("data/sample_policy.pdf")
output_path.parent.mkdir(parents=True, exist_ok=True)

with pymupdf.open() as document:
    page1 = document.new_page()
    page1.insert_text(
        (72, 72),
        "Employee Leave Policy\n"
        "Employees receive 20 days of annual leave."
    )

    page2 = document.new_page()
    page2.insert_text(
        (72, 72),
        "Remote Work Policy\n"
        "Employees may work remotely with manager approval."
    )

    document.save(output_path)

print(f"Created sample PDF: {output_path}")
