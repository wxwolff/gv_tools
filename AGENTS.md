# GV Tools project instructions

- Treat `/Users/dwolff/Desktop/Work/GV Tools` as the workspace for every task in this project.
- The package source and release workspace is `/Users/dwolff/Desktop/Work/GV Tools/gv_tools`.
- Build fresh release artifacts into `/Users/dwolff/Desktop/Work/GV Tools/gv_tools/release`.
- For every new version, update `docs/conf.py`, rebuild the complete Sphinx HTML
  manual in `docs/_build/html`, verify the generated title contains the new
  version, and include the rebuilt HTML in the source archive.
- Before delivering a release, remove older wheel and source-archive versions from that release directory so it contains only the current version.
- Do not use `/Users/dwolff/Documents/ChatGPT/GV Tools` as the working source tree.
- For every feature or workflow change, update the canonical master notebook at `/Users/dwolff/Desktop/Work/GV Tools/GV_Tools_Ingest_Demonstration.ipynb` and keep the package example copy synchronized.
- Keep the canonical sample-data tree at `/Users/dwolff/Desktop/Work/GV Tools/Samples`; do not create or package a duplicate `gv_tools/Samples` directory.
