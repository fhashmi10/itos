??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02unknown8ܝ
?
encoder/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*%
shared_nameencoder/dense/kernel

(encoder/dense/kernel/Read/ReadVariableOpReadVariableOpencoder/dense/kernel* 
_output_shapes
:
??*
dtype0
}
encoder/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*#
shared_nameencoder/dense/bias
v
&encoder/dense/bias/Read/ReadVariableOpReadVariableOpencoder/dense/bias*
_output_shapes	
:?*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
m
	dense
	variables
regularization_losses
trainable_variables
	keras_api

signatures
h

kernel
bias
		variables

regularization_losses
trainable_variables
	keras_api

0
1
 

0
1
?
	variables
regularization_losses
layer_metrics
metrics
trainable_variables
non_trainable_variables
layer_regularization_losses

layers
 
QO
VARIABLE_VALUEencoder/dense/kernel'dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUEencoder/dense/bias%dense/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?
		variables
layer_metrics

regularization_losses
metrics
trainable_variables
non_trainable_variables
layer_regularization_losses

layers
 
 
 
 

0
 
 
 
 
 
?
serving_default_input_1Placeholder*,
_output_shapes
:?????????@?*
dtype0*!
shape:?????????@?
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1encoder/dense/kernelencoder/dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_19603
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename(encoder/dense/kernel/Read/ReadVariableOp&encoder/dense/bias/Read/ReadVariableOpConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_19751
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameencoder/dense/kernelencoder/dense/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_19767??
?	
?
B__inference_encoder_layer_call_and_return_conditional_losses_19556
features
dense_19549:
??
dense_19551:	?
identity??dense/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallfeaturesdense_19549dense_19551*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_195482
dense/StatefulPartitionedCalls
ReluRelu&dense/StatefulPartitionedCall:output:0*
T0*,
_output_shapes
:?????????@?2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????@?2

Identityn
NoOpNoOp^dense/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?: : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:V R
,
_output_shapes
:?????????@?
"
_user_specified_name
features
?+
?
 __inference__wrapped_model_19511
input_1C
/encoder_dense_tensordot_readvariableop_resource:
??<
-encoder_dense_biasadd_readvariableop_resource:	?
identity??$encoder/dense/BiasAdd/ReadVariableOp?&encoder/dense/Tensordot/ReadVariableOp?
&encoder/dense/Tensordot/ReadVariableOpReadVariableOp/encoder_dense_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02(
&encoder/dense/Tensordot/ReadVariableOp?
encoder/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
encoder/dense/Tensordot/axes?
encoder/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
encoder/dense/Tensordot/freeu
encoder/dense/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
:2
encoder/dense/Tensordot/Shape?
%encoder/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%encoder/dense/Tensordot/GatherV2/axis?
 encoder/dense/Tensordot/GatherV2GatherV2&encoder/dense/Tensordot/Shape:output:0%encoder/dense/Tensordot/free:output:0.encoder/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2"
 encoder/dense/Tensordot/GatherV2?
'encoder/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'encoder/dense/Tensordot/GatherV2_1/axis?
"encoder/dense/Tensordot/GatherV2_1GatherV2&encoder/dense/Tensordot/Shape:output:0%encoder/dense/Tensordot/axes:output:00encoder/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"encoder/dense/Tensordot/GatherV2_1?
encoder/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
encoder/dense/Tensordot/Const?
encoder/dense/Tensordot/ProdProd)encoder/dense/Tensordot/GatherV2:output:0&encoder/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
encoder/dense/Tensordot/Prod?
encoder/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2!
encoder/dense/Tensordot/Const_1?
encoder/dense/Tensordot/Prod_1Prod+encoder/dense/Tensordot/GatherV2_1:output:0(encoder/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2 
encoder/dense/Tensordot/Prod_1?
#encoder/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#encoder/dense/Tensordot/concat/axis?
encoder/dense/Tensordot/concatConcatV2%encoder/dense/Tensordot/free:output:0%encoder/dense/Tensordot/axes:output:0,encoder/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2 
encoder/dense/Tensordot/concat?
encoder/dense/Tensordot/stackPack%encoder/dense/Tensordot/Prod:output:0'encoder/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
encoder/dense/Tensordot/stack?
!encoder/dense/Tensordot/transpose	Transposeinput_1'encoder/dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????@?2#
!encoder/dense/Tensordot/transpose?
encoder/dense/Tensordot/ReshapeReshape%encoder/dense/Tensordot/transpose:y:0&encoder/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2!
encoder/dense/Tensordot/Reshape?
encoder/dense/Tensordot/MatMulMatMul(encoder/dense/Tensordot/Reshape:output:0.encoder/dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2 
encoder/dense/Tensordot/MatMul?
encoder/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2!
encoder/dense/Tensordot/Const_2?
%encoder/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%encoder/dense/Tensordot/concat_1/axis?
 encoder/dense/Tensordot/concat_1ConcatV2)encoder/dense/Tensordot/GatherV2:output:0(encoder/dense/Tensordot/Const_2:output:0.encoder/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2"
 encoder/dense/Tensordot/concat_1?
encoder/dense/TensordotReshape(encoder/dense/Tensordot/MatMul:product:0)encoder/dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????@?2
encoder/dense/Tensordot?
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$encoder/dense/BiasAdd/ReadVariableOp?
encoder/dense/BiasAddBiasAdd encoder/dense/Tensordot:output:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????@?2
encoder/dense/BiasAdd{
encoder/ReluReluencoder/dense/BiasAdd:output:0*
T0*,
_output_shapes
:?????????@?2
encoder/Reluz
IdentityIdentityencoder/Relu:activations:0^NoOp*
T0*,
_output_shapes
:?????????@?2

Identity?
NoOpNoOp%^encoder/dense/BiasAdd/ReadVariableOp'^encoder/dense/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?: : 2L
$encoder/dense/BiasAdd/ReadVariableOp$encoder/dense/BiasAdd/ReadVariableOp2P
&encoder/dense/Tensordot/ReadVariableOp&encoder/dense/Tensordot/ReadVariableOp:U Q
,
_output_shapes
:?????????@?
!
_user_specified_name	input_1
?%
?
B__inference_encoder_layer_call_and_return_conditional_losses_19634
features;
'dense_tensordot_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freef
dense/Tensordot/ShapeShapefeatures*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transposefeaturesdense/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????@?2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/Tensordot/MatMul}
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????@?2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????@?2
dense/BiasAddc
ReluReludense/BiasAdd:output:0*
T0*,
_output_shapes
:?????????@?2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????@?2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp:V R
,
_output_shapes
:?????????@?
"
_user_specified_name
features
?
?
!__inference__traced_restore_19767
file_prefix9
%assignvariableop_encoder_dense_kernel:
??4
%assignvariableop_1_encoder_dense_bias:	?

identity_3??AssignVariableOp?AssignVariableOp_1?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
valuexBvB'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0* 
_output_shapes
:::*
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp%assignvariableop_encoder_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp%assignvariableop_1_encoder_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_2Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_2c

Identity_3IdentityIdentity_2:output:0^NoOp_1*
T0*
_output_shapes
: 2

Identity_3z
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"!

identity_3Identity_3:output:0*
_input_shapes
: : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
__inference__traced_save_19751
file_prefix3
/savev2_encoder_dense_kernel_read_readvariableop1
-savev2_encoder_dense_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
valuexBvB'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0/savev2_encoder_dense_kernel_read_readvariableop-savev2_encoder_dense_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0**
_input_shapes
: :
??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:

_output_shapes
: 
? 
?
@__inference_dense_layer_call_and_return_conditional_losses_19548

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:?????????@?2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????@?2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????@?2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:?????????@?2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:?????????@?
 
_user_specified_nameinputs
? 
?
@__inference_dense_layer_call_and_return_conditional_losses_19713

inputs5
!tensordot_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*,
_output_shapes
:?????????@?2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
Tensordot/MatMulq
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????@?2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????@?2	
BiasAddp
IdentityIdentityBiasAdd:output:0^NoOp*
T0*,
_output_shapes
:?????????@?2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:T P
,
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?
?
%__inference_dense_layer_call_fn_19722

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_195482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????@?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?: : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
,
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?%
?
B__inference_encoder_layer_call_and_return_conditional_losses_19665
input_1;
'dense_tensordot_readvariableop_resource:
??4
%dense_biasadd_readvariableop_resource:	?
identity??dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/freee
dense/Tensordot/ShapeShapeinput_1*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transposeinput_1dense/Tensordot/concat:output:0*
T0*,
_output_shapes
:?????????@?2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense/Tensordot/MatMul}
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:?2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*,
_output_shapes
:?????????@?2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*,
_output_shapes
:?????????@?2
dense/BiasAddc
ReluReludense/BiasAdd:output:0*
T0*,
_output_shapes
:?????????@?2
Relur
IdentityIdentityRelu:activations:0^NoOp*
T0*,
_output_shapes
:?????????@?2

Identity?
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?: : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp:U Q
,
_output_shapes
:?????????@?
!
_user_specified_name	input_1
?
?
'__inference_encoder_layer_call_fn_19674
input_1
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_195562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????@?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????@?
!
_user_specified_name	input_1
?
?
#__inference_signature_wrapper_19603
input_1
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_195112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????@?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?: : 22
StatefulPartitionedCallStatefulPartitionedCall:U Q
,
_output_shapes
:?????????@?
!
_user_specified_name	input_1
?
?
'__inference_encoder_layer_call_fn_19683
features
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallfeaturesunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:?????????@?*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_195562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*,
_output_shapes
:?????????@?2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@?: : 22
StatefulPartitionedCallStatefulPartitionedCall:V R
,
_output_shapes
:?????????@?
"
_user_specified_name
features"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
@
input_15
serving_default_input_1:0?????????@?A
output_15
StatefulPartitionedCall:0?????????@?tensorflow/serving/predict:?"
?
	dense
	variables
regularization_losses
trainable_variables
	keras_api

signatures
*&call_and_return_all_conditional_losses
_default_save_signature
__call__"
_tf_keras_model
?

kernel
bias
		variables

regularization_losses
trainable_variables
	keras_api
*&call_and_return_all_conditional_losses
__call__"
_tf_keras_layer
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
	variables
regularization_losses
layer_metrics
metrics
trainable_variables
non_trainable_variables
layer_regularization_losses

layers
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
,
serving_default"
signature_map
(:&
??2encoder/dense/kernel
!:?2encoder/dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
		variables
layer_metrics

regularization_losses
metrics
trainable_variables
non_trainable_variables
layer_regularization_losses

layers
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
B__inference_encoder_layer_call_and_return_conditional_losses_19634
B__inference_encoder_layer_call_and_return_conditional_losses_19665?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
 __inference__wrapped_model_19511input_1"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_encoder_layer_call_fn_19674
'__inference_encoder_layer_call_fn_19683?
???
FullArgSpec
args?
jself

jfeatures
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_dense_layer_call_and_return_conditional_losses_19713?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_dense_layer_call_fn_19722?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_19603input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_19511u5?2
+?(
&?#
input_1?????????@?
? "8?5
3
output_1'?$
output_1?????????@??
@__inference_dense_layer_call_and_return_conditional_losses_19713f4?1
*?'
%?"
inputs?????????@?
? "*?'
 ?
0?????????@?
? ?
%__inference_dense_layer_call_fn_19722Y4?1
*?'
%?"
inputs?????????@?
? "??????????@??
B__inference_encoder_layer_call_and_return_conditional_losses_19634h6?3
,?)
'?$
features?????????@?
? "*?'
 ?
0?????????@?
? ?
B__inference_encoder_layer_call_and_return_conditional_losses_19665g5?2
+?(
&?#
input_1?????????@?
? "*?'
 ?
0?????????@?
? ?
'__inference_encoder_layer_call_fn_19674Z5?2
+?(
&?#
input_1?????????@?
? "??????????@??
'__inference_encoder_layer_call_fn_19683[6?3
,?)
'?$
features?????????@?
? "??????????@??
#__inference_signature_wrapper_19603?@?=
? 
6?3
1
input_1&?#
input_1?????????@?"8?5
3
output_1'?$
output_1?????????@?