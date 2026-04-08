# Security policy

`moire_metrology` is a research package maintained by a single person. It
has no network surface, no authentication, and no production deployment
story — security issues here are mostly limited to things like unsafe file
parsing or dependency vulnerabilities.

## Reporting a vulnerability

If you believe you've found a security issue, please **do not** open a
public GitHub issue. Instead, report it privately by either:

- Using GitHub's private vulnerability reporting on this repository
  (Security tab → "Report a vulnerability"), or
- Emailing the maintainer directly (see the `authors` field in
  `CITATION.cff` or the package metadata in `pyproject.toml` for contact).

Please include:

- A description of the issue and its impact.
- Steps to reproduce, ideally a minimal script.
- The package version (`python -c "import moire_metrology; print(moire_metrology.__version__)"`).

I'll acknowledge the report as soon as I can and discuss next steps. Since
this is a single-maintainer research project, response times are best
effort — please be patient.

## Supported versions

Only the latest released version on `main` is supported. There are no
backports to older versions.
