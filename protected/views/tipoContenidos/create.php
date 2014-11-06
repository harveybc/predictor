<?php
$this->breadcrumbs=array(
	'Tipo de Contenidos'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista Tipo Contenidos', 'url'=>array('index')),
	array('label'=>'Gestionar Tipo Contenidos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Crear Tipo de Contenido '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'tipo-contenidos-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_form', array(
	'model'=>$model,
	'form' =>$form
	)); ?>

<div class="row buttons forms100c">
	<?php echo CHtml::submitButton(Yii::t('app', 'Aceptar')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
