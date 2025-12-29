ARG LAB_BASE_IMAGE=isaac-gr00t:r38.3.arm64-sbsa-cu130-24.04-jupyterlab_latest
FROM ${LAB_BASE_IMAGE}

# Optional: install course-specific Python deps on top of the gr00t base image.
# Keep this controlled via a build arg so we don't mutate the base image and can
# opt-in to the "delta" when needed.
ARG INSTALL_COURSE_DEPS=0

WORKDIR /workspace

COPY requirements-course.txt /tmp/requirements-course.txt

RUN if [ "${INSTALL_COURSE_DEPS}" = "1" ]; then \
      python3 -m pip install -U pip && \
      python3 -m pip install -r /tmp/requirements-course.txt && \
      rm -f /tmp/requirements-course.txt ; \
    else \
      echo "Skipping course deps install (INSTALL_COURSE_DEPS=${INSTALL_COURSE_DEPS})"; \
    fi
