"""Generate synthetic proof-of-address PNGs for local app testing."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).parent / "test_docs"


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_bill(
    path: Path,
    *,
    title: str,
    name: str,
    address: str,
    account: str,
    amount: str,
) -> None:
    width, height = 850, 1100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title_font = load_font(34)
    header_font = load_font(22)
    body_font = load_font(20)

    draw.rectangle((0, 0, width, 90), fill=(0, 82, 155))
    draw.text((40, 28), title, fill="white", font=title_font)

    y = 130
    lines = [
        ("Statement Date", "June 1, 2026"),
        ("Account Holder", name),
        ("Service Address", address),
        ("Account Number", account),
        ("Amount Due", amount),
    ]
    for label, value in lines:
        draw.text((40, y), label, fill=(80, 80, 80), font=header_font)
        draw.text((40, y + 28), value, fill="black", font=body_font)
        y += 90

    draw.line((40, y, width - 40, y), fill=(200, 200, 200), width=2)
    y += 30
    draw.text((40, y), "This is a synthetic test document for demo purposes only.", fill=(120, 120, 120), font=body_font)

    image.save(path)


def draw_invalid_doc(path: Path) -> None:
    width, height = 850, 1100
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(40)
    body_font = load_font(24)

    draw.text((40, 80), "20% OFF PIZZA", fill=(180, 30, 30), font=title_font)
    draw.text((40, 160), "Show this coupon at checkout.", fill="black", font=body_font)
    draw.text((40, 220), "Not valid as proof of address.", fill=(100, 100, 100), font=body_font)

    image.save(path)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    draw_bill(
        OUTPUT_DIR / "01_utility_bill_match.png",
        title="City Power & Light",
        name="Jane M. Doe",
        address="123 Main St, Indianapolis, IN 46204",
        account="ACC-987654",
        amount="$142.18",
    )
    draw_bill(
        OUTPUT_DIR / "02_name_mismatch.png",
        title="City Power & Light",
        name="John Q. Public",
        address="123 Main St, Indianapolis, IN 46204",
        account="ACC-112233",
        amount="$98.40",
    )
    draw_bill(
        OUTPUT_DIR / "03_address_mismatch.png",
        title="City Power & Light",
        name="Jane M. Doe",
        address="456 Oak Ave, Chicago, IL 60601",
        account="ACC-445566",
        amount="$76.22",
    )
    draw_bill(
        OUTPUT_DIR / "04_bank_statement_match.png",
        title="First Demo Bank",
        name="Jane M. Doe",
        address="123 Main St, Indianapolis, IN 46204",
        account="****4321",
        amount="$2,410.55",
    )
    draw_invalid_doc(OUTPUT_DIR / "05_invalid_coupon.png")

    print(f"Generated test PNGs in {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
