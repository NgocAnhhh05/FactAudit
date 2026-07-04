"""
Visualize the FACT-AUDIT Graph Architecture

This script compiles the master graph and exports it as a Mermaid PNG image.
Ensure you have Graphviz/Mermaid dependencies installed if required by LangGraph.
"""

import os
from main_graph import master_graph

def save_graph_image():
    print("Đang render sơ đồ kiến trúc LangGraph FACT-AUDIT...")
    
    try:
        # Tạo thư mục img nếu chưa có
        os.makedirs("img", exist_ok=True)
        
        # Lấy byte ảnh từ đồ thị
        # draw_mermaid_png() hỗ trợ vẽ cả Sub-graphs cực kỳ trực quan
        png_bytes = master_graph.get_graph(xray=True).draw_mermaid_png()
        
        output_filename = "img/fact_audit_architecture.png"
        with open(output_filename, "wb") as f:
            f.write(png_bytes)
            
        print(f"✅ Đã lưu sơ đồ luồng FACT-AUDIT thành công tại: {output_filename}")
        print("Hãy mở ảnh lên để chiêm ngưỡng siêu phẩm của bạn!")
        
    except Exception as e:
        print(f"❌ Có lỗi xảy ra khi render ảnh (Có thể thiếu thư viện graphviz/pygraphviz): {e}")

if __name__ == "__main__":
    save_graph_image()