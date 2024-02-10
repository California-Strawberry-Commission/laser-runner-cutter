#!/bin/bash -ex
# Helper script to bootstrap Farm_ng's app venv directories


bootstrap() {
    local application="${1}"
    local venv="${2:-venv}"
    local bin="${venv}/bin/activate"

    # Run apt install if required.
    # You must inspect for errors in the override script at this time.
    if [[ -f ${application}/apt_requirements.txt ]]
    then
        f_size=$(wc -c <"${application}/apt_requirements.txt")
        if [f_size -ge 1 ]
        then
            xargs sudo apt install -y <${application}/apt_requirements.txt || true
        fi
    fi
    PYTHON=${venv}/bin/python

    if [[ -f ${application}/setup.cfg.md5sum ]]
    then
        old_checksum=$(cat ${application}/setup.cfg.md5sum)
        new_checksum=$(md5sum ${application}/setup.cfg)
        if [ "$new_checksum" == "$old_checksum" ]
        then
            echo "Not updating venv"
        else
            rm ${venv}/.lock || true
            echo "setup.cfg md5sum changed, will update venv"
        fi
    fi

    if [ ! -f ${venv}/.lock ]; then
        echo "creating venv ${venv}"
        python3 -m venv ${venv}
        # Upgrade Pip
        $PYTHON -m pip install --upgrade pip
        $PYTHON -m pip install setuptools wheel
        cd ${application}
        $PYTHON -m pip install -e .
        touch ${venv}/.lock
        md5sum ${application}/setup.cfg > ${application}/setup.cfg.md5sum

    else
        echo "${venv} already existed, launching application"
    fi


}

bootstrap $1 $2
