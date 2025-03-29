#!/usr/bin/env python3

import argparse
import subprocess
import sys
import os
import time
import json
import shutil
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class TestResult:
    name: str
    status: str
    duration: float
    output: str
    error: Optional[str] = None

class TestRunner:
    def __init__(self, args):
        self.args = args
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
        # Ensure we're in the project root
        self.project_root = Path(__file__).parent.parent.absolute()
        os.chdir(self.project_root)
        
        # Check dependencies
        self.check_dependencies()

    def check_dependencies(self):
        """Check if required dependencies are installed"""
        print("\n🔍 Checking dependencies...")
        
        missing_deps = []
        install_instructions = []
        
        # Check Python packages
        python_deps = ["numpy"]
        for dep in python_deps:
            try:
                __import__(dep)
            except ImportError:
                missing_deps.append(f"Python package: {dep}")
                install_instructions.append(f"Install {dep}: pip install {dep}")
        
        # Check for MPI if distributed tests are enabled
        if not self.args.skip_distributed:
            if not shutil.which('mpirun'):
                if sys.platform == 'darwin':
                    missing_deps.append("MPI")
                    install_instructions.append("Install MPI: brew install openmpi")
                else:
                    missing_deps.append("MPI")
                    install_instructions.append("Install MPI: apt-get install openmpi-bin openmpi-common libopenmpi-dev")
            
            # Check for mlx.launch
            if not shutil.which('mlx.launch'):
                missing_deps.append("MLX development installation")
                install_instructions.append("Install MLX in development mode: pip install -e .")
        
        # Check for cmake
        if not shutil.which('cmake'):
            if sys.platform == 'darwin':
                missing_deps.append("CMake")
                install_instructions.append("Install CMake: brew install cmake")
            else:
                missing_deps.append("CMake")
                install_instructions.append("Install CMake: apt-get install cmake")
        
        if missing_deps:
            print("\n⚠️  Missing dependencies:")
            for dep in missing_deps:
                print(f"   - {dep}")
            print("\n📝 Installation instructions:")
            for instruction in install_instructions:
                print(f"   $ {instruction}")
            print("\nPlease install the missing dependencies and try again.")
            sys.exit(1)
        
        print("✅ All required dependencies found")

    def run_cpp_tests(self) -> List[TestResult]:
        """Run C++ tests using doctest framework"""
        print("\n🔍 Running C++ tests...")
        
        build_dir = self.project_root / "build"
        if not (build_dir / "tests" / "tests").exists():
            print("⚙️  Building C++ tests...")
            try:
                subprocess.run([
                    "cmake", 
                    "-B", "build",
                    "-DMLX_BUILD_METAL=OFF" if self.args.cpu_only else "",
                    "-DCMAKE_BUILD_TYPE=Debug" if self.args.debug else "-DCMAKE_BUILD_TYPE=Release"
                ], check=True)
                subprocess.run(["cmake", "--build", "build"], check=True)
            except subprocess.CalledProcessError as e:
                return [TestResult(
                    name="cpp_tests_build",
                    status="ERROR",
                    duration=0.0,
                    output="",
                    error=f"Failed to build C++ tests: {str(e)}"
                )]

        start = time.time()
        try:
            result = subprocess.run(
                ["./build/tests/tests", "--reporters=xml", "--out=test_results.xml"],
                capture_output=True,
                text=True
            )
            duration = time.time() - start
            
            return [TestResult(
                name="cpp_tests",
                status="PASSED" if result.returncode == 0 else "FAILED",
                duration=duration,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else None
            )]
        except subprocess.CalledProcessError as e:
            return [TestResult(
                name="cpp_tests",
                status="ERROR",
                duration=time.time() - start,
                output=e.output,
                error=str(e)
            )]

    def run_python_tests(self) -> List[TestResult]:
        """Run Python tests with various configurations"""
        print("\n🐍 Running Python tests...")
        
        configs = [
            {
                "name": "CPU Tests",
                "env": {"DEVICE": "cpu", "LOW_MEMORY": "1"},
            }
        ]
        
        if not self.args.cpu_only:
            configs.append({
                "name": "GPU Tests",
                "env": {
                    "DEVICE": "gpu",
                    "METAL_DEVICE_WRAPPER_TYPE": "1",
                    "METAL_DEBUG_ERROR_MODE": "0"
                },
            })

        results = []
        for config in configs:
            start = time.time()
            env = os.environ.copy()
            env.update(config["env"])
            
            # Ensure PYTHONPATH includes the mlx directory
            pythonpath = str(self.project_root / "mlx")
            env["PYTHONPATH"] = f"{pythonpath}:{env.get('PYTHONPATH', '')}"
            
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "unittest", "discover", "python/tests", "-v"],
                    env=env,
                    capture_output=True,
                    text=True
                )
                duration = time.time() - start
                
                results.append(TestResult(
                    name=config["name"],
                    status="PASSED" if result.returncode == 0 else "FAILED",
                    duration=duration,
                    output=result.stdout,
                    error=result.stderr if result.returncode != 0 else None
                ))
            except subprocess.CalledProcessError as e:
                results.append(TestResult(
                    name=config["name"],
                    status="ERROR",
                    duration=time.time() - start,
                    output=e.output,
                    error=str(e)
                ))

        return results

    def run_distributed_tests(self) -> List[TestResult]:
        """Run distributed tests"""
        print("\n🌐 Running distributed tests...")
        
        # Check if mlx.launch is available
        if not shutil.which('mlx.launch'):
            return [TestResult(
                name="distributed_tests",
                status="ERROR",
                duration=0.0,
                output="",
                error="mlx.launch not found. Please install MLX in development mode with: pip install -e ."
            )]
        
        tests = [
            {
                "name": "MPI Distributed Test",
                "cmd": ["mpirun", "--bind-to", "none", "-np", "8", 
                       sys.executable, "python/tests/mpi_test_distributed.py"]
            },
            {
                "name": "Ring Distributed Test",
                "cmd": ["mlx.launch", "--verbose", "-n", "8",
                       "python/tests/ring_test_distributed.py"]
            }
        ]
        
        results = []
        for test in tests:
            start = time.time()
            try:
                result = subprocess.run(
                    test["cmd"],
                    capture_output=True,
                    text=True
                )
                duration = time.time() - start
                
                results.append(TestResult(
                    name=test["name"],
                    status="PASSED" if result.returncode == 0 else "FAILED",
                    duration=duration,
                    output=result.stdout,
                    error=result.stderr if result.returncode != 0 else None
                ))
            except subprocess.CalledProcessError as e:
                results.append(TestResult(
                    name=test["name"],
                    status="ERROR",
                    duration=time.time() - start,
                    output=e.output,
                    error=str(e)
                ))
            except FileNotFoundError as e:
                results.append(TestResult(
                    name=test["name"],
                    status="ERROR",
                    duration=time.time() - start,
                    output="",
                    error=str(e)
                ))

        return results

    def write_safe(self, text):
        """Safely write text to stdout with retries"""
        while text:
            try:
                n = sys.stdout.write(text)
                sys.stdout.flush()
                text = text[n:]
            except BlockingIOError:
                time.sleep(0.1)  # Wait a bit before retrying
            except IOError:
                break  # Give up on fatal errors

    def generate_report(self):
        """Generate a detailed test report"""
        total_duration = time.time() - self.start_time
        passed = sum(1 for r in self.results if r.status == "PASSED")
        failed = sum(1 for r in self.results if r.status == "FAILED")
        errors = sum(1 for r in self.results if r.status == "ERROR")
        
        # Print summary
        summary = f"""
📊 Test Summary:
{"="*50}
Total Duration: {total_duration:.2f}s
Total Tests: {len(self.results)}
Passed: {passed} 🟢
Failed: {failed} 🔴
Errors: {errors} ⚠️
{"="*50}
"""
        self.write_safe(summary)
        
        # Print failures and errors
        if failed or errors:
            self.write_safe("\n❌ Failures and Errors:\n")
            for result in self.results:
                if result.status in ["FAILED", "ERROR"]:
                    self.write_safe(f"\n{result.name} - {result.status}\n")
                    if result.error:
                        self.write_safe(result.error + "\n")
                    if result.output:
                        self.write_safe(result.output + "\n")
        
        # Save JSON report
        report = {
            "summary": {
                "timestamp": datetime.now().isoformat(),
                "total_duration": total_duration,
                "total_tests": len(self.results),
                "passed": passed,
                "failed": failed,
                "errors": errors
            },
            "results": [
                {
                    "name": r.name,
                    "status": r.status,
                    "duration": r.duration,
                    "output": r.output,
                    "error": r.error
                } for r in self.results
            ]
        }
        
        try:
            with open("test_report.json", "w") as f:
                json.dump(report, f, indent=2)
        except IOError as e:
            self.write_safe(f"\nWarning: Could not save JSON report: {e}\n")

    def run(self):
        """Run all tests"""
        try:
            self.results.extend(self.run_cpp_tests())
            self.results.extend(self.run_python_tests())
            if not self.args.skip_distributed:
                self.results.extend(self.run_distributed_tests())
            
            self.generate_report()
            
            # Exit with appropriate status code
            failed_or_errors = any(r.status in ["FAILED", "ERROR"] for r in self.results)
            sys.exit(1 if failed_or_errors else 0)
        
        except KeyboardInterrupt:
            self.write_safe("\n⚠️  Test run interrupted by user\n")
            sys.exit(130)
        except Exception as e:
            self.write_safe(f"\n💥 Unexpected error: {e}\n")
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="MLX Test Runner")
    parser.add_argument("--cpu-only", action="store_true", help="Run only CPU tests")
    parser.add_argument("--debug", action="store_true", help="Build in debug mode")
    parser.add_argument("--skip-distributed", action="store_true", help="Skip distributed tests")
    
    args = parser.parse_args()
    
    print("🚀 Starting MLX Test Suite")
    runner = TestRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
