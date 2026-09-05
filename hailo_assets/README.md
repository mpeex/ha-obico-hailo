# hailo_assets

Questa cartella contiene i binari HailoRT in modo da poter costruire l'addon
**offline**, senza scaricare nulla a build-time.

## Convenzione di naming

I nomi dipendono da `BUILD_ARCH`, riepilogato dal `Dockerfile`:

| Arch        | `.deb`                            | `.whl`                                 |
|-------------|-----------------------------------|----------------------------------------|
| `aarch64`   | `hailort_${HAILORT_VERSION}_arm64.deb` | `hailort-${HAILORT_VERSION}-cp311-cp311-linux_aarch64.whl` |
| `amd64`     | `hailort_${HAILORT_VERSION}_amd64.deb` | `hailort-${HAILORT_VERSION}-cp311-cp311-linux_x86_64.whl`  |

Esempi (HailoRT 4.21.0 su Python 3.11):

- aarch64: `hailort_4.21.0_arm64.deb`, `hailort-4.21.0-cp311-cp311-linux_aarch64.whl`
- amd64: `hailort_4.21.0_amd64.deb`, `hailort-4.21.0-cp311-cp311-linux_x86_64.whl`

## Comportamento della build

Il `Dockerfile` (unico, self-contained) cerca in questa cartella i binari che
corrispondono a `BUILD_ARCH` e alla versione richiesta (`HAILORT_VERSION`):

- **se sono presenti** → la build li usa direttamente (nessun download, build offline);
- **se sono assenti** → la build li scarica da
  `https://dev-public.hailo.ai/${HAILORT_RELEASE}/` (`--fail`) e procede comunque
  (build online).

Questa cartella resta quasi sempre vuota a meno che non si voglia garantire una
build offline. I binari sono grandi e **non** vanno committati (sono in
`.gitignore`): vengono distribuiti **solo all'interno della docker image**.

## Licenze di redistribuzione

La redistribuzione dei binari HailoRT è consentita e richiede di includere le
relative licenze:

- `licenses/LICENSE-HailoRT-MIT.txt` — per libhailort, pyhailort e hailortcli (MIT);
- `licenses/LICENSE-HailoRT-LGPL-2.1.txt` — per il plugin GStreamer hailonet (LGPL-2.1-or-later).

Copie dei testi ufficiali dal repository [hailo-ai/hailort](https://github.com/hailo-ai/hailort)
(branch `hailo8`). Il `Dockerfile` le copia dentro l'immagine in
`/usr/share/licenses/hailort/`.
