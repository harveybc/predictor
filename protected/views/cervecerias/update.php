<?php
$this->breadcrumbs=array(
	'Cervecerias'=>array('index'),
	$model->descripcion=>array('view','id'=>$model->descripcion),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Cervecerias', 'url'=>array('index')),
	array('label'=>'Nueva Cerveceria', 'url'=>array('create')),
	array('label'=>'Detalles Cerveceria', 'url'=>array('view', 'id'=>$model->descripcion)),
	array('label'=>'Gestionar Cervecerias', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Cervecerias #<?php echo $model->descripcion; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'cervecerias-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Actualizar')); ?>
</div>

<?php $this->endWidget(); ?>

</div><!-- form -->
