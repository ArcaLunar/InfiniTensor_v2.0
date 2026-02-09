#pragma once
#ifndef PYTHON_DTYPE_HPP
#define PYTHON_DTYPE_HPP
#include "core/dtype.h"
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <torch/torch.h>

namespace py = pybind11;

namespace infini {
void bind_data_type(py::module &m) {
    py::class_<DataType>(m, "DType")
        // Constructor
        .def(py::init<infiniDtype_t>())

        // Methods
        .def("get_size", &DataType::getSize, "Get the size in bytes")
        .def("get_type", &DataType::getType, "Get the underlying enum type")
        .def("to_string", &DataType::toString, "Get string representation")

        // Properties (exposed through methods)
        .def_property_readonly("size", &DataType::getSize)
        .def_property_readonly("type", &DataType::getType)
        .def_property_readonly("name", &DataType::toString)

        // Operator overloading
        .def(py::self == py::self)
        .def(py::self != py::self);
}
inline DataType dtype_from_string(const std::string &dtype_str) {
    // Create DataType from string
    static std::unordered_map<std::string, infiniDtype_t> str_to_dtype = {
        {"byte", INFINI_DTYPE_BYTE},
        {"bool", INFINI_DTYPE_BOOL},
        {"torch.bool", INFINI_DTYPE_BOOL},
        {"int8", INFINI_DTYPE_I8},
        {"torch.int8", INFINI_DTYPE_I8},
        {"int16", INFINI_DTYPE_I16},
        {"torch.int16", INFINI_DTYPE_I16},
        {"torch.short", INFINI_DTYPE_I16},
        {"int32", INFINI_DTYPE_I32},
        {"torch.int32", INFINI_DTYPE_I32},
        {"torch.int", INFINI_DTYPE_I32},
        {"int64", INFINI_DTYPE_I64},
        {"torch.int64", INFINI_DTYPE_I64},
        {"torch.long", INFINI_DTYPE_I64},
        {"uint8", INFINI_DTYPE_U8},
        {"torch.uint8", INFINI_DTYPE_U8},
        {"uint16", INFINI_DTYPE_U16},
        {"torch.uint16", INFINI_DTYPE_U16},
        {"uint32", INFINI_DTYPE_U32},
        {"torch.uint32", INFINI_DTYPE_U16},
        {"uint64", INFINI_DTYPE_U64},
        {"torch.uint64", INFINI_DTYPE_U64},
        {"fp8", INFINI_DTYPE_F8},
        {"float16", INFINI_DTYPE_F16},
        {"torch.float16", INFINI_DTYPE_F16},
        {"torch.half", INFINI_DTYPE_F16},
        {"float32", INFINI_DTYPE_F32},
        {"torch.float32", INFINI_DTYPE_F32},
        {"torch.float", INFINI_DTYPE_F32},
        {"double", INFINI_DTYPE_F64},
        {"float64", INFINI_DTYPE_F64},
        {"torch.float64", INFINI_DTYPE_F64},
        {"torch.double", INFINI_DTYPE_F64},
        {"complex32", INFINI_DTYPE_C32},
        {"torch.complex32", INFINI_DTYPE_C32},
        {"torch.chalf", INFINI_DTYPE_C64},
        {"complex64", INFINI_DTYPE_C64},
        {"torch.complex64", INFINI_DTYPE_C64},
        {"torch.cfloat", INFINI_DTYPE_C64},
        {"complex128", INFINI_DTYPE_C128},
        {"torch.complex128", INFINI_DTYPE_C128},
        {"torch.cdouble", INFINI_DTYPE_C128},
        {"bfloat16", INFINI_DTYPE_BF16}};

    auto it = str_to_dtype.find(dtype_str);
    if (it != str_to_dtype.end()) {
        return DataType(it->second);
    }
    throw std::runtime_error("Unknown data type: " + dtype_str);
}

inline std::string dtype_to_string(const DataType &dtype) {
    // Create string from DataType
    static std::unordered_map<infiniDtype_t, std::string> dtype_to_str = {
        {INFINI_DTYPE_BYTE, "byte"},       {INFINI_DTYPE_BOOL, "bool"},
        {INFINI_DTYPE_I8, "int8"},         {INFINI_DTYPE_I16, "int16"},
        {INFINI_DTYPE_I32, "int32"},       {INFINI_DTYPE_I64, "int64"},
        {INFINI_DTYPE_U8, "uint8"},        {INFINI_DTYPE_U16, "uint16"},
        {INFINI_DTYPE_U32, "uint32"},      {INFINI_DTYPE_U64, "uint64"},
        {INFINI_DTYPE_F8, "fp8"},          {INFINI_DTYPE_F16, "float16"},
        {INFINI_DTYPE_F32, "float32"},     {INFINI_DTYPE_F64, "float64"},
        {INFINI_DTYPE_C32, "complex32"},   {INFINI_DTYPE_C64, "complex64"},
        {INFINI_DTYPE_C128, "complex128"}, {INFINI_DTYPE_BF16, "bfloat16"}};

    auto it = dtype_to_str.find(dtype.getType());
    if (it != dtype_to_str.end()) {
        return it->second;
    }
    throw std::runtime_error("Unknown data type: " +
                             std::to_string(dtype.getType()));
}

void bind_dtype_functions(py::module &m) {
    m.def("dtype_from_string", &dtype_from_string,
          "Create DataType from string", py::arg("dtype_str"))
        .def("dtype_to_string", &dtype_to_string, "Convert DataType to string",
             py::arg("dtype"));
}
} // namespace infini

#endif // PYTHON_DTYPE_HPP
