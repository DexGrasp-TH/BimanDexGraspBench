import os
import shutil


def resolve_symlinks_to_new_dir(src_root, dst_root):
    for root, dirs, files in os.walk(src_root):
        # 计算在新目录下应该创建的对应路径
        relative_path = os.path.relpath(root, src_root)
        new_root = os.path.join(dst_root, relative_path)
        os.makedirs(new_root, exist_ok=True)

        for name in files:
            src_path = os.path.join(root, name)
            dst_path = os.path.join(new_root, name)

            if os.path.islink(src_path):
                # 获取真实目标路径
                target = os.readlink(src_path)

                # 处理相对路径
                if not os.path.isabs(target):
                    target = os.path.join(root, target)

                if os.path.exists(target):
                    print(f"[LINK] {src_path} -> {target}")
                    # 复制原文件到新目录对应位置
                    shutil.copy2(target, dst_path)
                else:
                    print(f"[BROKEN] {src_path} -> {target} (target missing)")
            else:
                # 直接复制非软链接文件
                print(f"[FILE] {src_path} -> {dst_path}")
                shutil.copy2(src_path, dst_path)


if __name__ == "__main__":
    # src = input("Enter source folder: ").strip()
    # dst = input("Enter target folder: ").strip()
    src = "output/dataset_0_dual_ur5_shadow/succgrasp"
    dst = "output/dataset_0_dual_ur5_shadow/dataset"
    resolve_symlinks_to_new_dir(src, dst)
