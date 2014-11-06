<?php
$this->breadcrumbs=array(
	'Unidades Documentales'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Unidades', 'url'=>array('index')),
	array('label'=>'Gestionar Unidades', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Crear Unidad Documental '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'documentos-form',
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
