<?php
$this->layout='column1';
$this->breadcrumbs=array(
	'Documentos'=>array(Yii::t('app', 'index')),
	Yii::t('app', 'Crear'),
);

$this->menu=array(
	array('label'=>'Lista de Documentos', 'url'=>array('index')),
	array('label'=>'Gestionar Documentos', 'url'=>array('admin')),
);

// si el parámetro query está seteado, preselecciona el motor con el tag o nombre indicado.
    if  (isset($_GET['query']))
    {
        $modeloId=Motores::model()->findByAttributes(array("TAG"=>$_GET['query']));
    }
    else
    {
        $modeloId="H";
    }

?>

<?php $this->setPageTitle (' Subir Documento Electrónico de un Motor'); ?>
<div class="form">

<?php 
    //echo $_GET['query'];

    $form=$this->beginWidget('CActiveForm', array(
	'id'=>'documentos-form',
	'enableAjaxValidation'=>true,
    // Parámetro necesario para SUBIR archivos
    'htmlOptions'=>array('enctype' => 'multipart/form-data'),
)); 
echo $this->renderPartial('_formSubirMotor', array(
	'model'=>$model,
        'modelMeta'=>$modelMeta,
        'modelArchivo'=>$modelArchivo,
        'modeloId'=>$modeloId,
	'form' =>$form,
	)); ?>

<div class="row buttons">
	<?php echo CHtml::submitButton(Yii::t('app', 'Aceptar')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
