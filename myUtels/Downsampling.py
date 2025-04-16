import struct

def analyze_binary_ply(file_path):
    with open(file_path, 'rb') as f:
        header = []
        while True:
            line = f.readline().decode('utf-8').strip()
            header.append(line)
            if line == 'end_header':
                break

        format_line = [line for line in header if line.startswith('format')][0]
        is_little_endian = 'binary_little_endian' in format_line

        # Get property names and their types
        properties = []
        for line in header:
            if line.startswith('property'):
                parts = line.split()
                dtype, name = parts[1], parts[2]
                properties.append((dtype, name))
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[-1])

        print("Properties:", [p[1] for p in properties])

        # Define struct format
        fmt_map = {
            'float': 'f',
            'float32': 'f',
            'uchar': 'B',
            'uint8': 'B',
            'int': 'i',
            'int32': 'i',
            'ushort': 'H',
            'uint16': 'H',
            'double': 'd'
        }

        struct_fmt = ''.join(fmt_map[p[0]] for p in properties)
        struct_size = struct.calcsize(struct_fmt)
        endian_char = '<' if is_little_endian else '>'

        # Read and parse first 200 vertices
        data = []
        for _ in range(min(200, num_vertices)):
            binary_data = f.read(struct_size)
            if len(binary_data) < struct_size:
                break
            values = struct.unpack(endian_char + struct_fmt, binary_data)
            data.append(values)a

        # Transpose and analyze
        data_cols = list(zip(*data))
        for name, col in zip([p[1] for p in properties], data_cols):
            unique_vals = set(col)
            print(f"{name}: {len(unique_vals)} unique values → {list(unique_vals)[:10]}{'...' if len(unique_vals) > 10 else ''}")

# Example usage:
analyze_binary_ply(r"C:\Farshid\Uni\Semesters\Thesis\Data\DALESObjects\DALESObjects\train\5190_54400_new.ply")
