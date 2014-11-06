<?php
$this->breadcrumbs=array(
	'Competencias'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista de Competencias', 'url'=>array('index')),
	array('label'=>'Nueva Competencia', 'url'=>array('create')),
	array('label'=>'Detalles Competencia', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Competencias', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Competencias #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'competencias-form',
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
