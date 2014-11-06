<?php
$this->breadcrumbs=array(
	'Tipo de Contenidos'=>array('index'),
	$model->id=>array('view','id'=>$model->id),
	Yii::t('app', 'Actualizar'),
);

$this->menu=array(
	array('label'=>'Lista Tipo Contenidos', 'url'=>array('index')),
	array('label'=>'Nuevo Tipo Contenido', 'url'=>array('create')),
	array('label'=>'Detalles Tipo Contenido', 'url'=>array('view', 'id'=>$model->id)),
	array('label'=>'Gestionar Tipo Contenidos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Actualizar Tipo de Contenido #<?php echo $model->id; ?> '); ?>
<div class="form"><style>    .forms50cr{  float:left;    }</style>

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'tipo-contenidos-form',
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
