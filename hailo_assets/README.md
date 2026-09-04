# hailo_assets

Questa cartella contiene i binari HailoRT in modo da poter costruire l'addon
**offline**, senza scaricare nulla a build-time.

## Convenzione di naming

| Tipo | Nome atteso |
|------|-------------|
| Librerie runtime (`.deb`) | `hailort_${HAILORT_VERSION}_arm64.deb` |
| Binding Python (`.whl`)   | `hailort-${HAILORT_VERSION}-${PY_TAG}-${ARCH_TAG}.whl` |

Esempio (HailoRT 4.20.0 su Python 3.11 / aarch64):

- `hailort_4.20.0_arm64.deb`
- `hailort-4.20.0-cp311-cp311-linux_aarch64.whl`

## Comportamento della build

Il `Dockerfile` (unico, self-contained) cerca in questa cartella i binari con la
versione richiesta (`HAILORT_VERSION`, rilevata dal runtime host via
`hailortcli`):

- **se sono presenti** → la build li usa direttamente (nessun download, build offline);
- **se sono assenti** → la build li scarica da
  `https://dev-public.hailo.ai/${HAILORT_RELEASE}/` (`--fail`) e procede comunque
  (build online).

Questa cartella resta quasi sempre vuota a meno che non si voglia garantire una
build offline. I binari sono grandi e **non** vanno committati (sono in
`.gitignore`).
