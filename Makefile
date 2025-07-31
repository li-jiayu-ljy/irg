PY="python"
DEV="0"

# common
preprocess:
	${PY} datasets/${DATASET}/preprocess.py pre \
	  -d datasets/${DATASET}/data \
	  -o datasets/${DATASET}

run-sdv:
	-rm -r datasets/${DATASET}/out/sdv
	-mkdir -p datasets/${DATASET}/out/sdv
	date > datasets/${DATASET}/out/sdv/timing-log
	-${PY} datasets/${DATASET}/run_sdv.py \
	  -d datasets/${DATASET}/simplified \
	  -o datasets/${DATASET}/out/sdv \
	  -s ${SCALE}
	date >> datasets/${DATASET}/out/sdv/timing-log
	cp datasets/${DATASET}/out/sdv/metadata.json datasets/${DATASET}/schema/sdv.json
	test -e datasets/${DATASET}/out/sdv/generated && ${PY} datasets/${DATASET}/preprocess.py desimplify \
	  -d datasets/${DATASET}/out/sdv/generated

run-ind:
	-rm -r datasets/${DATASET}/out/ind
	${PY} datasets/${DATASET}/run_sdv.py \
	  -d datasets/${DATASET}/simplified \
	  -o datasets/${DATASET}/out/ind \
	  -m ind
	${PY} datasets/${DATASET}/preprocess.py desimplify \
	  -d datasets/${DATASET}/out/ind/generated

run-rctgan:
	-rm -r datasets/${DATASET}/out/rctgan
	cd baselines && pip install -e RCTGAN --no-deps --force-reinstall
	CUDA_VISIBLE_DEVICES=${DEV} ${PY} datasets/${DATASET}/run_rct.py \
	  -d datasets/${DATASET}/simplified \
	  -o datasets/${DATASET}/out/rctgan \
	  -s datasets/${DATASET}/schema/sdv.json
	cp datasets/${DATASET}/out/rctgan/metadata.json datasets/${DATASET}/schema/rctgan.json
	${PY} datasets/${DATASET}/preprocess.py desimplify \
	  -d datasets/${DATASET}/out/rctgan/generated

run-clava:
	-rm -r datasets/${DATASET}/out/clava
	mkdir -p datasets/${DATASET}/out/clava
	CUDA_VISIBLE_DEVICES=${DEV} ${PY} baselines/ClavaDDPM/process.py \
	  -s datasets/${DATASET}/schema/sdv.json \
	  -o datasets/${DATASET}/out/clava \
	  pre \
	  -d datasets/${DATASET}/simplified \
	  -n ${DATASET}
	mkdir -p datasets/${DATASET}/schema/clava
	cp datasets/${DATASET}/out/clava/data/*.json datasets/${DATASET}/schema/clava/
	${PY} baselines/ClavaDDPM/complex_pipeline.py --config_path datasets/${DATASET}/out/clava/config.json
	${PY} baselines/ClavaDDPM/process.py \
	  -s datasets/${DATASET}/schema/sdv.json \
	  -o datasets/${DATASET}/out/clava \
	  post
	${PY} datasets/${DATASET}/preprocess.py desimplify \
	  -d datasets/${DATASET}/out/clava/generated

run-irg:
	-rm -r datasets/${DATASET}/out/irg
	CUDA_VISIBLE_DEVICES=${DEV} ${PY} main.py \
	  -c  datasets/${DATASET}/schema/irg.yaml \
	  -i datasets/${DATASET}/preprocessed \
	  -o datasets/${DATASET}/out/irg

evaluate:
	${PY} datasets/${DATASET}/evaluate.py

# specific
football:
	make preprocess run-ind run-rctgan run-clava run-irg evaluate DATASET=football

bec:
	make preprocess run-ind run-rctgan run-clava run-irg evaluate DATASET=bec

smm:
	make preprocess run-sdv run-ind run-rctgan run-clava run-irg evaluate DATASET=smm SCALE=1