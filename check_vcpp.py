"""
Script kiểm tra Visual C++ Redistributable đã được cài đặt chưa
Chạy: python check_vcpp.py
"""

import winreg
import sys

def check_vcpp_installed():
    """Kiểm tra các phiên bản Visual C++ Redistributable đã cài"""
    
    print("="*70)
    print("KIỂM TRA VISUAL C++ REDISTRIBUTABLE")
    print("="*70)
    
    # Các registry paths cần kiểm tra
    registry_paths = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"),  # 2015-2022
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Classes\Installer\Dependencies"),
    ]
    
    found_versions = []
    
    # Kiểm tra registry
    for hkey, path in registry_paths:
        try:
            key = winreg.OpenKey(hkey, path)
            
            # Nếu là path Dependencies, duyệt qua các subkeys
            if "Dependencies" in path:
                i = 0
                while True:
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        if "VC" in subkey_name or "Microsoft Visual C++" in subkey_name:
                            subkey = winreg.OpenKey(key, subkey_name)
                            try:
                                display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                                version = winreg.QueryValueEx(subkey, "Version")[0]
                                found_versions.append((display_name, version))
                            except:
                                pass
                            winreg.CloseKey(subkey)
                        i += 1
                    except OSError:
                        break
            else:
                # Đọc version trực tiếp
                try:
                    version = winreg.QueryValueEx(key, "Version")[0]
                    major = winreg.QueryValueEx(key, "Major")[0]
                    minor = winreg.QueryValueEx(key, "Minor")[0]
                    found_versions.append((f"Visual C++ {major}.{minor}", version))
                except:
                    pass
            
            winreg.CloseKey(key)
        except FileNotFoundError:
            continue
        except Exception as e:
            continue
    
    # Hiển thị kết quả
    if found_versions:
        print("\n✅ ĐÃ CÀI ĐẶT các phiên bản sau:\n")
        seen = set()
        for name, version in found_versions:
            if name not in seen:
                print(f"  • {name}")
                print(f"    Version: {version}")
                seen.add(name)
        
        # Kiểm tra có phiên bản 2015-2022 không
        has_2015_2022 = any("2015" in str(name) or "2017" in str(name) or 
                           "2019" in str(name) or "2022" in str(name) 
                           for name, _ in found_versions)
        
        if has_2015_2022:
            print("\n✅ CÓ Visual C++ Redistributable 2015-2022 (cần cho TensorFlow)")
        else:
            print("\n⚠️  KHÔNG TÌM THẤY Visual C++ Redistributable 2015-2022")
            print("   TensorFlow CẦN phiên bản này!")
            
    else:
        print("\n❌ KHÔNG TÌM THẤY Visual C++ Redistributable nào!")
        print("   Bạn CẦN cài đặt để chạy TensorFlow")
    
    print("\n" + "="*70)
    print("HƯỚNG DẪN CÀI ĐẶT")
    print("="*70)
    print("\n1. Download từ Microsoft:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("\n2. Chạy file .exe và làm theo hướng dẫn")
    print("\n3. Restart máy tính sau khi cài")
    print("\n4. Chạy lại script này để kiểm tra")
    print("="*70)

def check_python_architecture():
    """Kiểm tra Python 32-bit hay 64-bit"""
    print("\n" + "="*70)
    print("THÔNG TIN PYTHON")
    print("="*70)
    
    import platform
    import struct
    
    bits = struct.calcsize("P") * 8
    print(f"\n  Python version: {sys.version}")
    print(f"  Architecture: {bits}-bit")
    print(f"  Platform: {platform.platform()}")
    
    if bits == 32:
        print("\n  ⚠️  BẠN ĐANG DÙNG PYTHON 32-BIT!")
        print("     TensorFlow chỉ hỗ trợ Python 64-bit trên Windows")
        print("     Cài lại Python 64-bit từ: https://www.python.org/downloads/")
    else:
        print("\n  ✅ Python 64-bit - OK!")

def check_tensorflow():
    """Thử import TensorFlow"""
    print("\n" + "="*70)
    print("KIỂM TRA TENSORFLOW")
    print("="*70)
    
    try:
        import tensorflow as tf
        print(f"\n✅ TensorFlow ĐÃ CÀI: version {tf.__version__}")
        print(f"   GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    except ImportError as e:
        print(f"\n❌ KHÔNG THỂ IMPORT TENSORFLOW!")
        print(f"   Lỗi: {str(e)}")
        
        if "DLL" in str(e):
            print("\n   → Lỗi DLL: Thiếu Visual C++ Redistributable")
            print("   → Giải pháp: Cài Visual C++ Redistributable 2015-2022")

if __name__ == "__main__":
    try:
        check_vcpp_installed()
        check_python_architecture()
        check_tensorflow()
        
        print("\n" + "="*70)
        print("HOÀN THÀNH KIỂM TRA")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ LỖI: {e}")
        import traceback
        traceback.print_exc()