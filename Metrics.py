original = 48 * 1024 * 1024   # 48MB in bytes
compressed = 41 * 1024 * 1024  # 41MB in bytes
lzma = 15 * 1024 * 1024
zlib = 19 * 1024 * 1024
ratio = original / compressed
bpp = (compressed * 8) / original
print(f'=== LSTM ===')
print(f'Compression ratio: {ratio:.3f}x')
print(f'Bits per byte:     {bpp:.4f}')
print(f'Space saving:      {(1 - compressed/original)*100:.1f}%')
print()
print(f'=== Baselines ===')
print(f'LZMA ratio: {original/lzma:.3f}x  bpp: {lzma*8/original:.4f}')
print(f'ZLIB ratio: {original/zlib:.3f}x  bpp: {zlib*8/original:.4f}')