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
// si el parámetro query está seteado, preselecciona el tablero con el tag o nombre indicado.
    if  (isset($_GET['query']))
    {
        $modeloId=Tableros::model()->findByAttributes(array("TAG"=>$_GET['query']));
    }
    // TODO: Quitar este else.
    else
    {
        $modeloId="H";
    }

?>

<?php $this->setPageTitle (' Subir Documento Electrónico de Motor'); ?>
<div class="form">

<?php $form=$this->beginWidget('CActiveForm', array(
	'id'=>'documentos-form',
	'enableAjaxValidation'=>true,
    // Parámetro necesario para SUBIR archivos
    'htmlOptions'=>array('enctype' => 'multipart/form-data'),
)); 
echo $this->renderPartial('_formSubirTablero', array(
	'model'=>$model,
        'modelMeta'=>$modelMeta,
        'modelArchivo'=>$modelArchivo,
	'modeloId'=>$modeloId,
	'form' =>$form
	)); ?>
    

<div class="row buttons">
	<?php echo CHtml::submitButton(Yii::t('app', 'Aceptar')); ?>
</div>

<?php $this->endWidget(); ?>

</div>
