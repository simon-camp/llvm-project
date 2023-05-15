// RUN: mlir-translate -mlir-to-cpp %s | FileCheck %s

emitc.struct.def !emitc.struct<"string_view",
  #emitc.member<"data" : !emitc.ptr<i8>>,
  #emitc.member<"size" : index>
>
// CHECK-LABEL: struct string_view {
// CHECK-NEXT: int8_t* data;
// CHECK-NEXT: size_t size;
// CHECK-NEXT: };

func.func @struct_arg(%arg0 : !emitc.struct<"string_view", #emitc.member<"data" : !emitc.ptr<i8>>, #emitc.member<"size" : index>>) {
  return
}
// CHECK-LABEL: void struct_arg(struct string_view v1)

func.func @struct_member_access(%arg0 : !emitc.struct<"string_view", #emitc.member<"data" : !emitc.ptr<i8>>, #emitc.member<"size" : index>>) {
  %0 = emitc.struct.member.read %arg0 <"size" : none> : (!emitc.struct<"string_view", #emitc.member<"data" : !emitc.ptr<i8>>, #emitc.member<"size" : index>>) -> index
  emitc.struct.member.write %0 %arg0 <"size" : none> : index, !emitc.struct<"string_view", <"data" : !emitc.ptr<i8>>, <"size" : index>>
  return
}

// CHECK-LABEL: void struct_member_access(struct string_view v1) {
// CHECK-NEXT: size_t v2 = v1.size;
// CHECK-NEXT: v1.size = v2;
// CHECK-NEXT: return;  
// CHECK-NEXT: }  
