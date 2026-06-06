from opentelemetry.trace import Status, StatusCode

from dlib import configure_tracing, get_tracer, shutdown_tracing


def divide(numerator: float, denominator: float) -> float:
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("example.calculation.divide") as span:
        span.set_attribute("calculation.numerator", numerator)
        span.set_attribute("calculation.denominator", denominator)

        try:
            result = numerator / denominator
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise

        span.set_attribute("calculation.result", result)
        span.set_status(Status(StatusCode.OK))
        return result


def main():
    configure_tracing("example_service")
    tracer = get_tracer(__name__)
    try:
        with tracer.start_as_current_span("example.service_lifecycle") as span:
            with tracer.start_as_current_span("example.service_startup"):
                pass

            result = divide(10, 2)
            span.set_attribute("service.calculation_result", result)

            with tracer.start_as_current_span("example.service_shutdown"):
                pass
    finally:
        shutdown_tracing()


if __name__ == "__main__":
    main()
