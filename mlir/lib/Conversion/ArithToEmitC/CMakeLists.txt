add_mlir_conversion_library(MLIRArithToEmitC
  ArithToEmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToEMitC

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIREmitCDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  )
