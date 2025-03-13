import os
import pathlib
from typing import Any, Dict

import setuptools  # type: ignore

here = pathlib.Path(__file__).parent.resolve()
about: Dict[Any, Any] = {}
with open(os.path.join(here, "meshagent", "agents", "version.py"), "r") as f:
    exec(f.read(), about)

setuptools.setup(
    name="meshagent-agents",
    version=about["__version__"],
    description="Agent Building Blocks for Meshagent",
    long_description=(here / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    url="",
    classifiers=[       
    ],
    keywords=[],
    license="Apache License 2.0",
    packages=setuptools.find_namespace_packages(include=[
        "meshagent.*",
    ]),
    python_requires=">=3.9.0",
    install_requires=[
        "pyjwt>=2.0.0",
        "pytest>=8.3.4",
        "pytest-asyncio>=0.24.0",        
        "meshagent-api>=0.0.4",
        "meshagent-tools>=0.0.4",
        "meshagent-openai>=0.0.4",
        "pydantic>=2.10.4",
        "pydantic-ai>=0.0.23",
        "chonkie>=0.5.1",
        "chonkie[semantic]>=0.5.1",
        "chonkie[openai]>=0.5.1"
    ],
    package_data={        
        "meshagent.agents": ["py.typed", "*.pyi", "**/*.pyi",  "**/*.js"],
    },
    project_urls={
        "Documentation": "https://meshagent.com",
        "Website": "https://meshagent.com",
        "Source": "https://github.com/meshagent",
    },
)
