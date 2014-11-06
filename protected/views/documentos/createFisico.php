<?php
$this->breadcrumbs=array(
	'Documentos'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Documentos', 'url'=>array('index')),
	array('label'=>'Gestionar Documentos', 'url'=>array('admin')),
);
?>

<?php $this->setPageTitle (' Ingresar Documento Físico '); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'documentos-form',
	'enableAjaxValidation'=>true,
)); 
echo $this->renderPartial('_formFisico', array(
	'model'=>$model,
        'modelMeta'=>$modelMeta,
	'form' =>$form
	)); ?>

<div class="row buttons">
	<?php echo CHtml::submitButton(Yii::t('app', 'Aceptar')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
