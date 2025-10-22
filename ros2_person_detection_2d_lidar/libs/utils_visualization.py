import colorsys
import hashlib


def id_to_color(id, saturation=0.75, value=0.95):
    hash_object = hashlib.sha256(str(id).encode())
    hash_digest = hash_object.hexdigest()

    hue = int(hash_digest[:16], 16) % 360 / 360.0

    rgb = colorsys.hsv_to_rgb(hue, saturation, value)

    return rgb
