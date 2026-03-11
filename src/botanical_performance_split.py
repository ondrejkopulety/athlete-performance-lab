"""Backwards-compatibility shim – real code lives in src/bo"""Backwards-compatibility shim – real co
_import os, sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))OT_PROJECT_ROOTatif _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROerf_analyzer im    sys.path.insert(0, _PROJECT_nafrom src.botanical.stops_detector iman
if __name__ == "__main__":
    from src.botanical.stops_detpro    from src.botanical.ste(    main()
