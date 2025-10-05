from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess
import os

class PostInstallCommand(install):
    """Post-installation: build C extensions and download weights."""
    
    def run(self):
        # 1. Build C extensions if present
        nms_path = os.path.join("PIPNet", "FaceBoxesV2", "utils", "nms")
        setup_path = os.path.join(nms_path, "setup.py")
        if os.path.exists(setup_path):
            print(f"Building C extensions in {nms_path}...")
            subprocess.check_call(["python", "setup.py", "build_ext", "--inplace"], cwd=nms_path)
        
        # 2. Run weight download script (Hugging Face)
        print("Downloading model weights via Hugging Face...")
        subprocess.check_call(["python", "-m", "PIPNet.download_weights"])
        
        # 3. Continue normal installation
        install.run(self)


with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="PIPNet",
    version="0.1",
    packages=find_packages(),
    install_requires=requirements,
    include_package_data=True,
    package_data={
        "PIPNet": ["data/**/*.txt"],  # meanface etc
    },
    cmdclass={
        'install': PostInstallCommand,
    },
)
