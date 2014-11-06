<?php
$this->breadcrumbs=array(
	'MotoresT'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Mediciones', 'url'=>array('index')),
	array('label'=>'Nueva Medición', 'url'=>array('create')),
	array('label'=>'Ver Mediciones', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Mediciones', 'url'=>array('admin')),
);
?>

<?php
$nombre="";
$modelTMP=$model;
if (isset($modelTMP->TAG))
        $nombre=$modelTMP->TAG;

?>
<?php $this->setPageTitle(' Actualizar medición:'.$nombre.''); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'termomotores-form',
	'enableAjaxValidation'=>true,
    'htmlOptions'=>array('enctype' => 'multipart/form-data'),
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
    'modelArchivo' => $modelArchivo,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Actualizar')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
